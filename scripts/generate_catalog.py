"""Generate a per-table Markdown schema catalog for one Postgres database.

For a given database this writes one Markdown file per table under
``<output-dir>/<database>/<table>.md``, each containing an (empty) description
placeholder, the table DDL (``CREATE TYPE`` enums it uses + ``CREATE TABLE``),
a Columns table joining live Postgres types with descriptions from the
dataset's ``<db>_column_meaning_base.json``, and a Foreign keys section.

For a whole-database run it also writes ``<output-dir>/<database>/_constraints.md``,
a single file listing every primary-key and foreign-key constraint in the db.

Examples
--------
Every benchmark database under ``--dataset-path`` — omit ``--database``::

    uv run python scripts/generate_catalog.py --output-dir catalogs

All tables of a single database on the lite container (:5432)::

    uv run python scripts/generate_catalog.py --database alien --output-dir catalogs

A single table, with verbose logging::

    uv run python scripts/generate_catalog.py \
        --database alien --table signals --output-dir catalogs -v

A database on the full container (:5433) — swap the port in the DSN template::

    uv run python scripts/generate_catalog.py \
        --database crypto --output-dir catalogs \
        --db-dsn-template "postgresql://root:123123@localhost:5433/{database}"

Point at a different dataset location for column descriptions::

    uv run python scripts/generate_catalog.py \
        --database alien --output-dir catalogs \
        --dataset-path data/bird_interact/bird-interact-full

Run with ``--help`` for the full list of options.
"""

from __future__ import annotations
from conversation2sql.eval_framework.agents.utils_kb_linearize import (
    build_kb_filenames,
    build_kb_overview,
    linearize_prerequisites,
)
from conversation2sql.eval_framework.dataset_readers.bird_interact_reader import (
    _get_external_knowledge,
)

import argparse
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

import psycopg2
from psycopg2.extensions import connection as PgConnection

from conversation2sql.eval_framework.dataset_readers.bird_interact_reader import (
    _get_column_meanings,
)
from extract_ddl import (
    Column,
    Table,
    _render_table_ddl,
    fetch_enums,
    fetch_tables,
    load_table,
    open_readonly,
    select_examples,
)

logger = logging.getLogger("generate_catalog")


# Curated, hand-written table descriptions, keyed by
# ``database -> {table_name: description}`` (table names lowercased to match
# Postgres' unquoted-identifier reporting). Each value states what the table
# is -- its grain, primary key and how it links to sibling tables -- the
# context that the per-column ``## Columns`` table cannot convey; it
# deliberately does NOT re-list every column. Injected into the
# ``## Description`` section of each table's Markdown file by
# ``render_table_markdown`` so it survives re-runs (live DB introspection has
# no notion of a human table description). Tables absent from this map render
# with an empty description.
DATABASE_DESCRIPTIONS: dict[str, str] = {
    'alien': 'SETI-style radio-signal detection and analysis platform. Telescopes at ground observatories detect signals; each signal carries a full measurement set and spawns one-to-one extensions for observational conditions, source properties, classification, decoding, dynamics, probabilities, advanced phenomena and research-workflow status.',
    'archeology': 'Archaeological 3D-scanning platform linking sites, projects, field personnel and equipment to per-scan records. Each scan record fans out into derived metric tables covering point cloud, mesh, spatial measurements, detected features, environmental conditions, post-processing, registration, QC and conservation assessment.',
    'credit': 'Credit-scoring system that decomposes a single scoring event into a chain of six 1:1 segments: core demographics and decision status, employment and income, expenses and assets, banking and KYC/AML, credit-bureau compliance, and credit-account history.',
    'cross_db': 'Cross-border data-transfer governance platform. Each data flow has associated records for data classification, risk management, privacy-law compliance, security controls, third-party vendor risk and an audit trail tying those records together.',
    'crypto': 'Cryptocurrency exchange platform covering the full order-to-execution lifecycle. Orders generate executions, fees and risk/margin records; account balances snapshot user wallets; market-data snapshots feed aggregate stats and analytics indicators that drive system-monitoring metrics.',
    'cybermarket': 'Darknet-market intelligence database. Markets host vendors who list products; buyers place transactions; each transaction links to communications/IP logs, fraud/AML risk analysis, security-monitoring events and investigation cases.',
    'disaster': 'Disaster-response logistics platform. Each disaster event spawns a relief operation tied to a distribution hub; further records capture supplies, transport fleets, beneficiary assessments, financials, HR staffing, environment and health, and coordination/evaluation outcomes.',
    'fake': 'Fake-account detection platform for social media. Account and profile records feed behavioral signals (session, content, network, messaging), which chain into device/technical fingerprints, security-detection scores and moderation/enforcement actions.',
    'gaming': 'Gaming-peripheral test platform. Each test session records device identity, sensor and click performance, mechanical switch specs, audio/media specs, input/control features, physical durability and RGB-lighting configuration.',
    'insider': 'Insider-trading surveillance platform. Trader master records link to per-trade-day blotters; each blotter entry drives advanced behavioral analysis, sentiment/fundamentals snapshots, compliance cases, investigation details and enforcement actions.',
    'mental': 'Mental-health clinical platform. Patient, clinician and facility master records anchor per-patient assessments (with social/diagnosis and symptom/risk extensions), clinical encounters, treatment plans and outcome records.',
    'museum': 'Museum-collection management platform. Artifact master records link to ratings, security/access, environmental sensitivity and risk assessments. Exhibition halls contain showcases with environmental reading records (air quality, light/radiation, surface/physical). Condition assessments, conservation records and usage records cross-link artifacts to halls and showcases.',
    'news': 'News-recommendation platform. User and article master records link through device, session, recommendation and interaction records; each interaction has a one-to-one metrics extension and system-performance record.',
    'polar': 'Polar-equipment telemetry platform for Arctic/Antarctic machinery. Each equipment unit at a location generates operation/maintenance, power/battery, communications, cabin-environment, engine/fluids, transmission, chassis/vehicle, lighting/safety, scientific-instrument, thermal/renewable/grid and water/waste records.',
    'robot': 'Industrial-robot telemetry platform. Robot master and detail records link to per-operation records; each operation fans out into actuation telemetry, joint condition, joint performance, mechanical status, system-controller metrics, performance/safety indices and maintenance/fault data.',
    'solar': 'Solar-farm monitoring platform. Plant and panel master records anchor electrical characterization, performance and environment records; inverter, maintenance and alert records tie back to plants and panels.',
    'vaccine': 'Vaccine cold-chain tracking platform. Each shipment contains containers and transport vehicles; vaccine batches and data-logger devices link to containers and shipments; sensor readings tie containers to vehicles; regulatory and maintenance records close the compliance loop.',
    'virtual': 'Virtual-idol fan-engagement platform. Fan and idol master records link through a central interaction record; fan-side records cover membership/spending, engagement, loyalty/achievements, commerce/collection, events/fan-club, social-community, moderation/compliance, preferences/settings, retention/influence, support/feedback and free-form notes.',
}

TABLE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    'alien': {
        'signals': 'Central fact table of detected radio signals (PK `signalregistry`), linked to the detecting telescope; holds the detection timestamp and the core RF measurement set.',
        'observationalconditions': "One-to-one extension of `signals` recording the observation's date, time and duration.",
        'observatories': 'Master record of ground observatory stations (PK `observstation`) and their observing/atmospheric conditions.',
        'telescopes': 'Master record of telescopes (PK `telescregistry`), each located at an observatory, with equipment status and subsystem health.',
        'researchprocess': 'Per-signal research-workflow status (analysis priority, review, publication, funding, disclosure).',
        'signaladvancedphenomena': 'Per-signal exotic-analysis flags (lensing, quantum effects, encryption, message content, significance).',
        'signalclassification': 'Per-signal classification results and computed complexity/entropy/confidence metrics.',
        'signaldecoding': 'Per-signal decoding-attempt details (encoding, method, status, confidence).',
        'signaldynamics': 'Per-signal stability and coherence ratings.',
        'signalprobabilities': 'Per-signal probability scores for false-positive and source origin (technological/biological/natural/artificial).',
        'sourceproperties': 'Per-signal properties of the candidate celestial source (sky position, distance, object type and physical parameters).',
    },
    'archeology': {
        'scans': 'Central scan record (PK `questregistry`) tying together a project, operator and site, with capture settings.',
        'sites': 'Master list of archaeological sites (PK `zoneregistry`): location, cultural period, preservation and access status.',
        'projects': 'Master list of archaeological projects (PK `arcregistry`): funding source and permit info.',
        'personnel': 'Master list of field operators (PK `crewregistry`) and their supervisors.',
        'equipment': 'Master list of scanning-equipment units (PK `equipregistry`): type, model, calibration and condition.',
        'scanpointcloud': 'Point-cloud capture metrics for a project scan.',
        'scanmesh': '3D mesh-generation output for a site scan.',
        'scanspatial': 'Computed spatial measurements (area, volume, bounding box, orientation) for a scan.',
        'scanfeatures': 'Detected-feature and artifact results for a site scan.',
        'scanenvironment': 'Environmental and positioning conditions during a site scan.',
        'scanprocessing': 'Post-processing workflow record (software, resource usage, storage, calibration).',
        'scanregistration': 'Scan-alignment (registration) log: accuracy, control points, method and error.',
        'scanqc': 'Quality-control record for a project scan (validation, archival, publication status).',
        'scanconservation': 'Conservation assessment of scanned structures (damage, priority, stability).',
    },
    'credit': {
        'core_record': 'Root credit-scoring record (PK `coreregistry`), one per scoring event; carries references, scoring/decision status and core demographics. Head of a 1:1 segment chain.',
        'employment_and_income': 'Per-applicant employment and income segment (1:1 with `core_record`).',
        'expenses_and_assets': 'Per-applicant expenses, assets and net-worth segment (incl. a JSONB housing block).',
        'bank_and_transactions': 'Per-applicant banking, insurance and KYC/AML identity-verification segment.',
        'credit_and_compliance': 'Per-applicant credit-bureau and compliance segment (screening, score, risk, derogatories).',
        'credit_accounts_and_history': 'Per-applicant credit-account profile (utilization, payment history, product usage).',
    },
    'cross_db': {
        'dataflow': 'Central register of cross-border data transfers (PK `recordregistry`): origin/destination, channel and volume.',
        'dataprofile': 'Data-classification profile for a flow (type, sensitivity, volume, validation).',
        'riskmanagement': 'Risk-management record for a flow (risk score, breaches, SLA, cost, maturity).',
        'compliance': 'Privacy-law compliance record for a flow (legal basis, consent, GDPR/CCPA/PIPL levels).',
        'securityprofile': 'Security-controls profile for a flow (encryption, access, logging, backup/DR).',
        'vendormanagement': 'Third-party vendor risk record (assessment, contracts, certification, monitoring).',
        'auditandcompliance': 'Audit-trail record linking compliance, data-profile and vendor records.',
    },
    'crypto': {
        'users': 'Master user accounts (PK `usersnode`) with an external UUID and account scope.',
        'orders': 'Central order book (PK `orderspivot`, unique `recordvault`): instrument, side, price/qty, status and timestamps.',
        'orderexecutions': 'Individual fills/executions for an order.',
        'fees': 'Per-order fee and maker-rebate record.',
        'accountbalances': 'Per-user balance snapshot (wallet/available/frozen/margin balances and PnL).',
        'marketdata': 'Market-microstructure snapshots (PK `marketdatanode`) held in a JSONB quote/depth block.',
        'marketstats': 'Aggregate market statistics for a snapshot (funding, open interest, 24h volume/price, supply, liquidity).',
        'analyticsindicators': 'Advanced market-analytics record (JSONB signal bundle) linking a snapshot and a stats row.',
        'riskandmargin': 'Per-order risk/margin profile (JSONB: leverage, liquidation, collateral).',
        'systemmonitoring': 'Platform-health metrics (API, latency, websocket, slippage) linked to an analytics record.',
    },
    'cybermarket': {
        'markets': 'Master list of darknet markets (PK `mktregistry`): size, activity and a JSONB reputation block.',
        'vendors': 'Vendor accounts (PK `vendregistry`) on a market: reputation, transaction history and verification tier.',
        'buyers': 'Buyer accounts (PK `buyregistry`): activity, spending pattern and risk rating; linked to a market/vendor.',
        'products': 'Product/service listings (PK `prodregistry`): category, price and quantity; linked to a vendor.',
        'transactions': 'Central transaction record (PK `txregistry`): payment, escrow, shipping and routing; linked to market/product/buyer.',
        'communication': 'Communication/log-analysis record (IP/Tor/VPN, fingerprints, suspicious-pattern scores).',
        'riskanalysis': 'Risk-analysis record (fraud/AML risk, wallet and transaction-chain metrics, JSONB network analytics).',
        'securitymonitoring': 'Security-monitoring record (audit, vulnerabilities, encryption, alerts, JSONB threat analysis).',
        'investigation': 'Investigation-case record (status, priority, escalation, action taken); links security and risk records.',
    },
    'disaster': {
        'disasterevents': 'Central register of disaster events (PK `distregistry`): hazard type/level, location and a JSONB impact block.',
        'operations': 'Central relief-operations record (PK `opsregistry`) tying an event to a hub; phase, status and priority.',
        'distributionhubs': 'Distribution-hub records for an event (PK `hubregistry`): capacity, storage and inventory metrics.',
        'supplies': 'Per-event/hub supplies record (JSONB inventory of food, water, shelter, medical and power).',
        'transportation': 'Per-event/hub transport-fleet and delivery-performance record.',
        'beneficiariesandassessments': 'Per-event/operation beneficiary-registration and needs-assessment record.',
        'financials': 'Per-event/operation budget, cost and funding record.',
        'humanresources': 'Per-event/operation staffing record (JSONB staffing profile).',
        'environmentandhealth': 'Per-event environment and public-health record.',
        'coordinationandevaluation': 'Per-event/operation coordination, evaluation and lessons-learned record.',
    },
    'fake': {
        'account': 'Master social-media account record (PK `accindex`): identifiers, type, age, status and verification.',
        'profile': 'Account profile record (JSONB composition of username, bio, location and contact credibility).',
        'sessionbehavior': 'Per-profile session/login-behavior record (incl. a JSONB activity-time distribution).',
        'contentbehavior': 'Per-session content-posting behavior metrics.',
        'networkmetrics': 'Per-session network-metrics record (JSONB connection/engagement/diversity block).',
        'messaginganalysis': 'Messaging-behavior analysis linking content and network metrics.',
        'technicalinfo': 'Technical/device-fingerprint record (IP, VPN/proxy/Tor, devices, user-agent) linking messaging and network.',
        'securitydetection': 'Security-detection event (JSONB detection-score profile) linking technical-info.',
        'moderationaction': 'Moderation/enforcement record (violations, cluster role, action taken) linking content and security records.',
    },
    'gaming': {
        'testsessions': 'Central test-session record (PK `sessionregistry`): device category, resource use, battery and latency timing.',
        'deviceidentity': 'Per-session device-identity record (make/model/firmware, connection, microcontroller, sensor).',
        'performance': 'Per-session sensor/click performance record.',
        'mechanical': 'Mechanical/switch specs (keys, switches, stabilizers, keycaps) linking device and performance.',
        'audioandmedia': 'Audio/media specs (sound, mic/speaker, Bluetooth) linking device and performance.',
        'interactionandcontrol': 'Input/control feature record (sensors, haptics, triggers, sticks) linking device and durability.',
        'physicaldurability': 'Build/durability record (weight, grip, friction, ingress resistance) linking performance and RGB.',
        'rgb': 'RGB-lighting record (brightness, modes, zones) linking mechanical and audio/media.',
    },
    'insider': {
        'trader': 'Master trader accounts (PK `tradereg`): type, account size, trading style and a JSONB performance/risk profile.',
        'transactionrecord': 'Per-trader trade-day blotter record (PK `transreg`): order behavior and a JSONB market-abuse indicators block.',
        'advancedbehavior': 'Advanced behavioral-analysis record (pattern/peer/market correlations) for a transaction.',
        'sentimentandfundamentals': 'Sentiment/fundamentals snapshot (news/social, ownership, options, leakage) for a transaction.',
        'compliancecase': 'Compliance-case record triggered by a transaction (screening, risk/alert level, detection).',
        'investigationdetails': 'Detailed investigation record expanding a compliance case (network analysis, evidence).',
        'enforcementactions': 'Enforcement-action record for a compliance case (penalty, restrictions, remediation).',
    },
    'mental': {
        'patients': 'Master patient record (PK `patkey`): demographics and social determinants; linked to a lead clinician.',
        'clinicians': 'Master clinician record (PK `clinkey`): confidence, documentation and follow-up plan; linked to a facility.',
        'facilities': 'Master facility record (PK `fackey`): referral source, stressors and a JSONB support-and-resources block.',
        'assessmentbasics': 'Core assessment record (PK `abkey`) for a patient: type, method, duration and validity.',
        'assessmentsocialanddiagnosis': 'One-to-one assessment extension covering social functioning and diagnosis.',
        'assessmentsymptomsandrisk': 'One-to-one assessment extension covering symptoms and risk (JSONB symptom scores).',
        'encounters': 'Clinical-encounter record (PK `enckey`) linking an assessment and patient.',
        'treatmentbasics': 'Per-encounter treatment record (medications, adherence, JSONB therapy details).',
        'treatmentoutcomes': 'Per-treatment outcome record (progress, response, satisfaction, engagement).',
    },
    'museum': {
        'artifactscore': 'Master artifact record (PK `artregistry`): name, era, age, material and conservation status.',
        'artifactratings': 'Per-artifact value ratings and conservation-difficulty assessment.',
        'artifactsecurityaccess': 'Per-artifact security, loan and documentation-status record.',
        'sensitivitydata': 'Per-artifact environmental/handling sensitivity profile.',
        'riskassessments': 'Per-artifact/hall risk-assessment record (risk level, evacuation, handling).',
        'conditionassessments': 'Condition-assessment record linking an artifact, showcase and light reading.',
        'conservationandmaintenance': 'Conservation/maintenance record linking an artifact, hall and surface reading.',
        'usagerecords': 'Per-artifact usage record (display rotation, handling and maintenance frequencies) linking showcase and sensitivity.',
        'exhibitionhalls': 'Exhibition-hall record (PK `hallrecord`): security and visitor-flow metrics.',
        'showcases': 'Showcase record (PK `showcasereg`) in a hall: sealing, climate-control and safety systems.',
        'environmentalreadingscore': 'Core environmental-reading record (PK `envreadregistry`) for a showcase (temperature, humidity, pressure).',
        'airqualityreadings': 'Air-quality readings for an environmental reading.',
        'lightandradiationreadings': 'Light/UV/IR radiation readings for an environmental reading.',
        'surfaceandphysicalreadings': 'Surface/physical condition readings for an environmental reading.',
    },
    'news': {
        'users': 'Master user record (PK `userkey`): subscription, segment, demographics and JSONB preferences.',
        'articles': 'Master article record (PK `artkey`): metadata, quality scores, format and a JSONB engagement-metrics block.',
        'devices': 'Per-user device record (OS/browser, screen, connection).',
        'sessions': 'User-session record (PK `seshkey`) linking user and device: engagement, geo and personalization metrics.',
        'recommendations': 'Recommendation records (PK `reckey`) for an article (algorithm, position, relevance scores).',
        'interactions': 'User-interaction events (PK `intkey`) linking a session and recommendation.',
        'interactionmetrics': 'One-to-one interaction extension (JSONB scroll/time/conversion metrics).',
        'systemperformance': 'System-performance record (response, load, cache, version) linking device and session.',
    },
    'polar': {
        'equipment': 'Master equipment register (PK `equipmentcode`): type, model and reliability/performance indices.',
        'location': 'Master location/station register (PK `locationregistry`): Arctic/Antarctic site and coordinates.',
        'operationmaintenance': 'Operation/maintenance record (hours, schedule, status, JSONB costs) linking equipment and location.',
        'powerbattery': 'Power/battery record for equipment (status, source, JSONB battery health).',
        'communication': 'Communication-system record (radio, antenna, JSONB signal metrics) linking equipment and location.',
        'cabinenvironment': 'Cabin-environment record (beacon, ventilation, doors, JSONB climate) linking equipment, location and comms.',
        'engineandfluids': 'Engine/fluids telemetry (speed, load, temperature, JSONB fluids) linking equipment and power.',
        'transmission': 'Transmission telemetry (temperature, pressure, gear) linking engine and equipment.',
        'chassisandvehicle': 'Chassis/vehicle telemetry (brakes, tracks, speed, JSONB tires) linking engine, equipment and transmission.',
        'lightingandsafety': 'Lighting/safety-system record (lighting, life-support, JSONB safety sensors) for equipment.',
        'scientific': 'Scientific-instrument status record for equipment.',
        'thermalsolarwindandgrid': 'Thermal/renewable/grid record (solar, wind, fuel cell, JSONB renewables) linking equipment, comms and power.',
        'waterandwaste': 'Water/waste-system record for equipment.',
        'weatherandstructure': 'Weather and structural-load record linking location and operation-maintenance.',
    },
    'robot': {
        'robot_record': 'Master robot record (PK `recreg`): record timestamp and robot code.',
        'robot_details': 'Per-robot specifications (PK `botdetreg`, 1:1 with `robot_record`): make, type, payload and reach.',
        'operation': 'Central operation record (PK `operreg`) linking robot-details and record: application, mode and cycle.',
        'actuation_data': 'Per-operation actuation telemetry (PK `actreg`): tool-center-point pose/speed, errors and per-motor current/voltage.',
        'joint_condition': 'Per-operation joint-condition telemetry (per-joint temperature, vibration, backlash).',
        'joint_performance': 'Per-operation joint-performance record (JSONB per-joint angle/speed/torque).',
        'mechanical_status': 'Per-operation mechanical-status telemetry (per-brake/encoder/gearbox status and readings).',
        'system_controller': 'System-controller record (JSONB controller load/memory/thermal metrics); PK references `actuation_data`.',
        'performance_and_safety': 'Performance/safety record (indices, energy, tool wear, JSONB safety metrics); PK references `actuation_data`.',
        'maintenance_and_fault': 'Maintenance/fault record (fault code, prediction, remaining life, cost); PK references `actuation_data`.',
    },
    'solar': {
        'plant': 'Master solar-plant register (PK `growregistry`, UUID): name, capacity and commissioning date.',
        'panel': 'Master solar-panel register (PK `panemark`) linked to a plant: make, type, rated power and efficiency.',
        'electrical': 'Per-panel electrical-characterization record (initial vs current Isc/Voc/Imp/Vmp, fill factor).',
        'performance': 'Per-panel performance record (measured power, loss, JSONB efficiency/degradation).',
        'environment': 'Per-plant-area environmental record (temperature, soiling, weather, JSONB irradiance).',
        'inverter': 'Per-plant inverter record (temperature, grid voltage/frequency, JSONB power metrics).',
        'maintenance': 'Maintenance record (inspection, schedule, warranty, costs) linking plant, panel and performance.',
        'alerts': 'Alert records (PK `alertreg`) linking plant, panel and performance: status and maintenance priorities.',
    },
    'vaccine': {
        'shipments': 'Central shipment register (PK `shipmentregistry`): customs, permits, integrity and tamper status.',
        'container': 'Shipping-container record (PK `containregistry`) linked to a shipment: coolant, power and status.',
        'transportinfo': 'Transport-vehicle record (PK `vehiclereg`): temperature, GPS and route; linked to shipment and container.',
        'vaccinedetails': 'Vaccine-batch record (PK `vacregistry`): variant, lot, expiry and dose counts; linked to container and shipment.',
        'datalogger': 'Data-logger device record (PK `loggerreg`): status, storage and sync; linked to container and shipment.',
        'sensordata': 'Sensor-reading record (temperature, shock/tilt/impact, alerts) linking container and vehicle.',
        'regulatoryandmaintenance': 'Regulatory/maintenance/inspection record linking shipment and vehicle.',
    },
    'virtual': {
        'fans': 'Master fan record (PK `userregistry`): tier, points, status and a JSONB personal-attributes block.',
        'virtualidols': 'Master virtual-idol register (PK `entityreg`): name, type, debut, genre and language.',
        'interactions': 'Central fan-idol interaction record (PK `activityreg`): action, platform, gifts and a JSONB engagement-metrics block.',
        'membershipandspending': 'Per-fan membership and spending record; linked to a fan.',
        'engagement': 'Per-fan engagement record (interaction frequency, content/language preference) linking interactions and membership.',
        'loyaltyandachievements': 'Per-fan loyalty/achievements record (rank, reputation, JSONB rewards) linking engagement and events.',
        'commerceandcollection': 'Per-fan merchandise/collection record linking engagement and membership.',
        'eventsandclub': 'Per-fan events/fan-club record (JSONB participation summary) linking membership and social-community.',
        'socialcommunity': 'Per-fan social-community record (JSONB community engagement) linking commerce and engagement.',
        'moderationandcompliance': 'Per-fan moderation/compliance record linking interactions and social-community.',
        'preferencesandsettings': 'Per-fan preferences/settings and usage-stats record linking membership and social-community.',
        'retentionandinfluence': 'Per-fan retention/influence record (churn, referrals, reach) linking engagement and loyalty.',
        'supportandfeedback': 'Per-fan support/feedback record (tickets, surveys, satisfaction, NPS) linking interactions and preferences.',
        'additionalnotes': 'Free-form notes about a fan, linked to a retention/influence record.',
    },
}


def _enums_used_by_table(
    columns: list[Column],
    all_enums: list[tuple[str, list[str]]],
) -> list[tuple[str, list[str]]]:
    enum_names = {name for name, _ in all_enums}
    used: set[str] = set()
    for col in columns:
        base = col.data_type.strip().removesuffix("[]").strip().strip('"')
        if base in enum_names:
            used.add(base)
    return [(name, labels) for name, labels in all_enums if name in used]


def load_column_meanings(dataset_path: Path, db_name: str) -> dict[str, dict[str, str]]:
    """Load `<db>_column_meaning_base.json` into {table: {column: meaning}}.

    Keys in the source file are lowercased `db|table|column`. Missing file
    yields an empty mapping (catalog still renders, with empty descriptions).
    """
    try:
        raw = _get_column_meanings(dataset_path, db_name)
    except FileNotFoundError:
        logger.warning(
            "no column-meaning file for %s under %s; descriptions will be empty",
            db_name,
            dataset_path,
        )
        return {}
    out: dict[str, dict[str, str]] = {}
    prefix = f"{db_name.lower()}|"
    for key, entry in raw.items():
        if not key.startswith(prefix):
            continue
        parts = key.split("|")
        if len(parts) != 3:
            continue
        _, table, column = parts
        out.setdefault(table, {})[column] = entry.column_meaning
    return out


def _md_cell(text: str) -> str:
    """Sanitize a value for a Markdown table cell (escape pipes, flatten newlines)."""
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def render_table_markdown(
    table: Table,
    enums: list[tuple[str, list[str]]],
    column_meanings: dict[str, str],
    referenced_by: list[str],
    examples_by_col: dict[str, list[str]],
    description: str = "",
) -> str:
    lines: list[str] = []
    lines.append(f"# table: {table.name}")
    lines.append("")
    lines.append("## Description")
    # Curated, hand-written table description from TABLE_DESCRIPTIONS (empty
    # string when none is registered, leaving the placeholder blank).
    lines.append(description)
    lines.append("")  # blank line before the next heading

    lines.append("## DDL")
    lines.append("```sql")
    for ename, labels in enums:
        quoted = ", ".join(f"'{lbl}'" for lbl in labels)
        lines.append(f'CREATE TYPE "{ename}" AS ENUM ({quoted});')
    if enums:
        lines.append("")
    # descriptions=None: column descriptions live in the ## Columns table
    # below, so the DDL block stays a clean CREATE TABLE.
    lines.append(_render_table_ddl(table, None, include_foreign_keys=True))
    lines.append("```")
    lines.append("")

    lines.append("## Columns")
    lines.append("| column | type | description | examples |")
    lines.append("| --- | --- | --- | --- |")
    for col in table.columns:
        name = _md_cell(col.name)
        data_type = _md_cell(col.data_type)
        desc = _md_cell(column_meanings.get(col.name.lower(), ""))
        examples = _md_cell(", ".join(examples_by_col.get(col.name, [])))
        lines.append(f"| {name} | {data_type} | {desc} | {examples} |")

    fk_lines = [
        f"- {fk.column} -> {fk.ref_table}({fk.ref_column})" for fk in table.foreign_keys
    ]
    ref_lines = [f"- referenced by: {ref}" for ref in referenced_by]
    # if fk_lines or ref_lines:
    #     lines.append("")
    #     lines.append("## Foreign keys")
    #     lines.extend(fk_lines)
    #     lines.extend(ref_lines)

    lines.append("")
    return "\n".join(lines)


def _primary_key_columns(table: Table) -> list[str]:
    """Primary-key column names for a table (composite or single-column)."""
    if table.composite_pk:
        return list(table.composite_pk)
    return [col.name for col in table.columns if col.is_pk]


def render_constraints_markdown(database: str, tables: list[Table]) -> str:
    """Render a single Markdown file listing every PK/FK constraint in the db.

    A ``## Primary keys`` fenced SQL block (one ``ALTER TABLE ... ADD
    CONSTRAINT ... PRIMARY KEY (...);`` statement per table that has a PK)
    followed by a ``## Foreign keys`` fenced SQL block (one ``ALTER TABLE ...
    ADD CONSTRAINT ... FOREIGN KEY (...) REFERENCES ...;`` statement per FK,
    no ``ON DELETE`` clause). Tables are listed in the order they were loaded.
    """
    lines: list[str] = []
    lines.append(f"# constraints: {database}")
    lines.append("")

    lines.append("## Primary keys")
    lines.append("```sql")
    for table in tables:
        pk_cols = _primary_key_columns(table)
        if not pk_cols:
            continue
        cols = ", ".join(f'"{c}"' for c in pk_cols)
        lines.append(
            f'ALTER TABLE "{table.name}" ADD CONSTRAINT "pk_{table.name}" '
            f"PRIMARY KEY ({cols});"
        )
    lines.append("```")
    lines.append("")

    lines.append("## Foreign keys")
    lines.append("```sql")
    for table in tables:
        for fk in table.foreign_keys:
            lines.append(
                f'ALTER TABLE "{table.name}" '
                f'ADD CONSTRAINT "fk_{table.name}_{fk.column}" '
                f'FOREIGN KEY ("{fk.column}") '
                f'REFERENCES "{fk.ref_table}" ("{fk.ref_column}");'
            )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def render_database_overview_markdown(
    database: str,
    table_names: list[str],
    external_kb: dict | None = None,
) -> str:
    """Render ``database_overview.md``: DB description + per-table bullet list
    + (when ``external_kb`` is given) a Knowledge Base index of every entry.

    ``external_kb`` here is the full, unmasked KB (this file is a DB-level,
    not task-level, artifact) — the deep_agent's per-task catalog dir
    re-renders this section from the task's masked KB at runtime instead of
    trusting this disk copy, so no masked entry actually reaches the agent.
    """
    lines: list[str] = []
    lines.append(f"# database: {database}")
    lines.append("")
    lines.append("## Description")
    lines.append(DATABASE_DESCRIPTIONS.get(database, ""))
    lines.append("")
    lines.append("## Tables")
    table_descs = TABLE_DESCRIPTIONS.get(database, {})
    for name in table_names:
        desc = table_descs.get(name.lower(), "")
        lines.append(f"- **{name}**: {desc}")
    if external_kb:
        filenames = build_kb_filenames(external_kb)
        kb_overview = build_kb_overview(external_kb, filenames)
        if kb_overview:
            lines.append("")
            lines.append("## Knowledge Base")
            lines.append(kb_overview)
    lines.append("")
    return "\n".join(lines)


def fetch_databases(dataset_path: Path) -> list[str]:
    """List benchmark databases under ``dataset_path``, sorted alphabetically.

    A database is a subdirectory holding ``<name>_column_meaning_base.json``
    (the per-DB marker BIRD-Interact ships); this skips ``.git`` and any stray
    directories.
    """
    return sorted(
        d.name
        for d in dataset_path.iterdir()
        if d.is_dir() and (d / f"{d.name}_column_meaning_base.json").is_file()
    )


def fetch_referenced_by(conn: PgConnection, schema: str, table: str) -> list[str]:
    """Tables/columns that hold a foreign key pointing AT ``table``."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c2.relname AS referencing_table,
                   a.attname  AS referencing_column
            FROM pg_constraint con
            JOIN pg_class c1     ON c1.oid = con.confrelid
            JOIN pg_class c2     ON c2.oid = con.conrelid
            JOIN pg_namespace n  ON n.oid = c1.relnamespace
            JOIN pg_attribute a  ON a.attrelid = con.conrelid
                                AND a.attnum = ANY(con.conkey)
            WHERE con.contype = 'f'
              AND c1.relname = %s
              AND n.nspname = %s
            ORDER BY referencing_table, referencing_column
            """,
            (table, schema),
        )
        return [f"{row[0]}({row[1]})" for row in cur.fetchall()]


def _generate_tables_for_db(dataset_path, database, db_out, conn, schema, only_table):
    meanings = load_column_meanings(dataset_path, database)
    all_enums = fetch_enums(conn, schema)
    table_names = [only_table] if only_table else fetch_tables(conn, schema)
    written = 0
    loaded: list[Table] = []
    for name in table_names:
        try:
            # Pass an empty DSN so load_table skips fetch_examples: the
            # catalog has no example-rows section, and fetch_examples in
            # extract_ddl currently mis-calls _format_result.
            table = load_table(conn, schema, name, "")
            enums = _enums_used_by_table(table.columns, all_enums)
            referenced_by = fetch_referenced_by(conn, schema, name)
            # meanings are keyed by lowercased table name; Postgres reports
            # unquoted identifiers lowercased, which is the BIRD-Interact norm.
            per_table = meanings.get(name.lower(), {})
            fk_by_col = {fk.column: fk for fk in table.foreign_keys}
            examples_by_col = {
                col.name: select_examples(conn, schema, table, col, fk_by_col)
                for col in table.columns
            }
            description = TABLE_DESCRIPTIONS.get(database, {}).get(name.lower(), "")
            md = render_table_markdown(
                table, enums, per_table, referenced_by, examples_by_col, description
            )
        except Exception:
            # Any per-table failure (DB error, introspection quirk, render
            # bug) skips that table and continues, so one bad table cannot
            # abort an unattended full-database run.
            logger.warning("skipping table %s", name, exc_info=True)
            continue
        (db_out / f"{name}.md").write_text(md, encoding="utf-8")
        loaded.append(table)
        written += 1
    logger.info("wrote %d table file(s) under %s", written, db_out)

    # Db-level constraints file: only meaningful for a whole-database run,
    # since a single-table run cannot list cross-table PK/FK relationships.
    if only_table is None and loaded:
        constraints_md = render_constraints_markdown(database, loaded)
        (db_out / "_foreign_key_constraints.md").write_text(
            constraints_md, encoding="utf-8"
        )
        logger.info("wrote constraints file %s", db_out / "_foreign_key_constraints.md")
    return written


def _generate_kb_for_db(dataset_path, database, kb_out):
    """
    Generate a knowledge base for a database.
    """

    external_kb = _get_external_knowledge(dataset_path, database)
    filenames = build_kb_filenames(external_kb)
    written = 0
    for kb_name in external_kb:
        kb_file = kb_out / f"{filenames[kb_name]}.md"
        kb_linearize_content = linearize_prerequisites(kb_name, external_kb)
        kb_file.write_text(kb_linearize_content, encoding="utf-8")
        written += 1

    return written


def generate_catalog_for_db(
    database: str,
    output_dir: Path,
    db_dsn_template: str,
    dataset_path: Path,
    schema: str = "public",
    only_table: str | None = None,
) -> int:
    dsn = db_dsn_template.format(database=database)
    parsed = urlparse(dsn)
    conn = open_readonly(
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        user=parsed.username or "root",
        password=parsed.password or "",
        dbname=database,
    )
    db_out = output_dir / database / "tables"
    db_out.mkdir(parents=True, exist_ok=True)
    kb_out = output_dir / database / "knowledge_base"
    kb_out.mkdir(parents=True, exist_ok=True)
    try:
        written_tbl = _generate_tables_for_db(
            dataset_path, database, db_out, conn, schema, only_table
        )
        written_kb = _generate_kb_for_db(dataset_path, database, kb_out)
        if only_table is None:
            table_names = fetch_tables(conn, schema)
            external_kb = _get_external_knowledge(dataset_path, database)
            overview_md = render_database_overview_markdown(database, table_names, external_kb)
            (output_dir / database / "database_overview.md").write_text(overview_md, encoding="utf-8")
            logger.info("wrote overview file %s", output_dir / database / "database_overview.md")
        logger.info(
            f"catalog generation for {database} complete: {written_tbl} table(s), {written_kb} knowledge base(s)",
        )
    except Exception:
        logger.error(f"catalog generation failed for {database}", exc_info=True)
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a per-table Markdown schema catalog for one database."
    )
    p.add_argument(
        "--database",
        default=None,
        help="Postgres database name (default: every database under --dataset-path)",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--table", default=None, help="single table (default: all tables)")
    p.add_argument(
        "--db-dsn-template",
        default="postgresql://root:123123@localhost:5432/{database}",
        help="DSN with a {database} placeholder; use :5433 for the full dataset",
    )
    p.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/bird_interact/bird-interact-lite"),
        help="locates <db>_column_meaning_base.json for column descriptions",
    )
    p.add_argument("--schema", default="public")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.table and not args.database:
        logger.error("--table requires --database (cannot target one table across all databases)")
        return 1
    try:
        databases = (
            [args.database]
            if args.database
            else fetch_databases(args.dataset_path)
        )
        if not args.database:
            logger.info("no --database given; generating catalog for %d database(s): %s",
                        len(databases), ", ".join(databases))
        for database in databases:
            generate_catalog_for_db(
                database=database,
                output_dir=args.output_dir,
                db_dsn_template=args.db_dsn_template,
                dataset_path=args.dataset_path,
                schema=args.schema,
                only_table=args.table,
            )
    except (psycopg2.Error, OSError, ValueError) as exc:
        # psycopg2.Error: connection/extraction; OSError: output write/path;
        # ValueError: malformed meaning JSON (json.JSONDecodeError). Report
        # cleanly with a non-zero exit instead of a raw traceback.
        logger.error("catalog generation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
