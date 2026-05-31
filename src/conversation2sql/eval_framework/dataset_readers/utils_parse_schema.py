import re


def _extract_tables_from_ddl_format(schema_string, table_names):
    """
    Extracts tables, sample data, foreign keys, and indexes.
    Filters FKs and Indexes to ensure they only reference the selected tables.
    """
    extracted_sections = []
    found_tables = []

    # 1. Extract Table Definitions and Sample Data
    table_pattern = (
        r'(CREATE TABLE\s+"({})"\s+.*?)(?=CREATE TABLE|-- Indexes|-- Foreign keys|$)'
    )

    for table in table_names:
        match = re.search(
            table_pattern.format(re.escape(table)),
            schema_string,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            extracted_sections.append(match.group(1).strip())
            # Capture the actual table name found (maintaining case)
            found_tables.append(match.group(2).lower())

    # 2. Extract and Filter Indexes
    # Matches: CREATE [UNIQUE] INDEX ... ON [schema].tablename ... ;
    index_matches = re.findall(
        r'(CREATE.*?INDEX.*?ON\s+(?:public\.)?(\w+|"[^"]+").*?;)',
        schema_string,
        re.IGNORECASE,
    )

    valid_indexes = []
    for full_index_sql, table_name in index_matches:
        # Strip quotes if present for the comparison
        clean_table_name = table_name.replace('"', "").lower()
        if clean_table_name in found_tables:
            valid_indexes.append(full_index_sql.strip())

    if valid_indexes:
        extracted_sections.append("-- Indexes\n" + "\n".join(valid_indexes))

    # 3. Extract and Filter Foreign Keys
    fk_section_match = re.search(
        r"-- Foreign keys(.*)", schema_string, re.DOTALL | re.IGNORECASE
    )
    if fk_section_match:
        fk_content = fk_section_match.group(1)
        # Captures: 1. Source Table, 2. Target Table
        fk_regex = r'(ALTER TABLE\s+"([^"]+)"\s+.*?REFERENCES\s+"([^"]+)"\s+.*?;)'
        all_fks = re.findall(fk_regex, fk_content, re.IGNORECASE | re.DOTALL)

        valid_fks = []
        for full_fk_sql, source_table, target_table in all_fks:
            if (
                source_table.lower() in found_tables
                and target_table.lower() in found_tables
            ):
                valid_fks.append(full_fk_sql.strip())

        if valid_fks:
            extracted_sections.append("-- Foreign keys\n" + "\n".join(valid_fks))

    return "\n\n".join(extracted_sections)


def _extract_tables_from_toon_format(schema_string, table_names):
    """
    Extracts tables, columns, foreign keys, and indexes from a TOON formatted schema.
    Filters foreign keys to ensure they only reference the selected tables, and
    updates the bracket counts for sections accordingly.
    """
    target_tables_lower = {t.lower() for t in table_names}
    found_tables = set()

    # Prepend a newline to ensure the regex catches the very first block if necessary.
    # Split the schema by table definition starts ("- name:").
    table_blocks_raw = re.split(r"\n\s*-\s*name:\s*", "\n" + schema_string)

    if len(table_blocks_raw) < 2:
        return ""  # No tables found in the schema

    # The first element is the preamble (database, dialect, tables[N]:, etc.)
    preamble = table_blocks_raw[0].strip()

    tables_data = {}

    # Identify found tables and store their data blocks
    for block in table_blocks_raw[1:]:
        lines = block.split("\n")
        if not lines:
            continue
        t_name = lines[0].strip()
        if t_name.lower() in target_tables_lower:
            found_tables.add(t_name.lower())
            tables_data[t_name.lower()] = block

    if not found_tables:
        return ""

    extracted_table_blocks = []

    # Process tables in the order requested by the user
    for t_name in table_names:
        t_name_lower = t_name.lower()
        if t_name_lower not in tables_data:
            continue

        block = tables_data[t_name_lower]
        lines = block.split("\n")
        new_lines = []

        in_fk_section = False
        fk_header_line = ""
        ref_table_idx = 1
        valid_fk_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()

            # 1. Detect the start of a foreign_keys section
            fk_match = re.match(
                r"^(\s*)foreign_keys\[\d+\]\{(.*?)\}:", line, re.IGNORECASE
            )
            if fk_match:
                in_fk_section = True
                fk_header_line = line
                # Dynamically find the index of 'ref_table' from the header definition
                fields = [f.strip().lower() for f in fk_match.group(2).split(",")]
                ref_table_idx = (
                    fields.index("ref_table") if "ref_table" in fields else 1
                )
                i += 1
                continue

            if in_fk_section:
                # 2. Check if we've hit a new section (e.g., "indexes[1]{...}:")
                if re.match(
                    r"^\s*\w+\[\d+\](?:\{.*?\})?:", line
                ) or stripped_line.startswith("- name:"):
                    in_fk_section = False

                    # Insert the valid foreign keys and update the block count
                    if valid_fk_lines:
                        new_header = re.sub(
                            r"foreign_keys\[\d+\]",
                            f"foreign_keys[{len(valid_fk_lines)}]",
                            fk_header_line,
                            flags=re.IGNORECASE,
                        )
                        new_lines.append(new_header)
                        new_lines.extend(valid_fk_lines)

                    # Append the new section header line
                    new_lines.append(line)

                elif stripped_line == "":
                    # Ignore empty lines inside the FK block to clean up formatting
                    pass
                else:
                    # 3. Parse and filter the foreign key data line
                    parts = line.split(",")
                    if len(parts) > ref_table_idx:
                        ref_table = parts[ref_table_idx].strip()
                        # Keep only if the target table is part of the extracted subset
                        if ref_table.lower() in found_tables:
                            valid_fk_lines.append(line)
            else:
                # Non-FK sections (columns, indexes) are passed through entirely
                new_lines.append(line)

            i += 1

        # Catch-all: If the block ended while still inside the FK section
        if in_fk_section and valid_fk_lines:
            new_header = re.sub(
                r"foreign_keys\[\d+\]",
                f"foreign_keys[{len(valid_fk_lines)}]",
                fk_header_line,
                flags=re.IGNORECASE,
            )
            new_lines.append(new_header)
            new_lines.extend(valid_fk_lines)

        # Reconstruct the specific table block (adding back the "- name: " prefix)
        reconstructed_block = "  - name: " + "\n".join(new_lines).rstrip()
        extracted_table_blocks.append(reconstructed_block)

    # Update the total table count in the preamble (e.g., tables[15]: -> tables[2]:)
    new_preamble = re.sub(r"tables\[\d+\]:", f"tables[{len(found_tables)}]:", preamble)

    # Combine the preamble and all extracted tables
    final_schema = new_preamble.rstrip() + "\n" + "\n".join(extracted_table_blocks)

    return final_schema
