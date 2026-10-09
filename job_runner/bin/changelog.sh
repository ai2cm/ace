#!/usr/bin/env bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHANGELOG_PATH="$SCRIPT_DIR/../CHANGELOG.md"

get_todos() {
    # Find the line number where ## TODO section starts
    local TODO_LINE
    TODO_LINE=$(grep -n "^## TODO" "$CHANGELOG_PATH" | cut -d: -f1)
    if [[ -z "$TODO_LINE" ]]; then
        echo "Error: ## TODO section not found in CHANGELOG.md" >&2
        exit 1
    fi
    # Extract lines after ## TODO, filter for lines starting with '-', and remove blank lines
    # Then remove the leading '- ' and number the lines
    sed -n "$((TODO_LINE + 1)),\$p" "$CHANGELOG_PATH" | \
        grep "^-" | \
        grep -v "^$" | \
        sed 's/^- //' | \
        awk '{print NR ". " $0}'
}

select_todo() {
    local todos=("$@")
    local num_todos=${#todos[@]}
    while true; do
        read -rp "Select a TODO number: " SELECTED_TODO_NUM
        # Validate input is a number
        if ! [[ "$SELECTED_TODO_NUM" =~ ^[0-9]+$ ]]; then
            echo "Error: Please enter a valid number" >&2
            continue
        fi
        # Validate number is in range
        if (( SELECTED_TODO_NUM < 1 || SELECTED_TODO_NUM > num_todos )); then
            echo "Error: Please enter a number between 1 and $num_todos" >&2
            continue
        fi
        # Valid input, get the selected todo
        local selected="${todos[$((SELECTED_TODO_NUM - 1))]}"
        # Remove the "N. " prefix and store in global variable
        SELECTED_TODO_TEXT=$(echo "$selected" | sed 's/^[0-9]*\. //')
        break
    done
}

categorize_todo() {
    local todo_num="$1"
    local todo_text="$2"
    local category_choice
    while true; do
        read -rp "Category [a)dded, c)hanged, f)ixed, o)ther TODO, q)uit]: " category_choice
        case "$category_choice" in
            a)
                echo "Moving TODO #$todo_num to 'Added'"
                SELECTED_CATEGORY="Added"
                return 0
                ;;
            c)
                echo "Moving TODO #$todo_num to 'Changed'"
                SELECTED_CATEGORY="Changed"
                return 0
                ;;
            f)
                echo "Moving TODO #$todo_num to 'Fixed'"
                SELECTED_CATEGORY="Fixed"
                return 0
                ;;
            o)
                # Return special code to indicate "select other TODO"
                return 1
                ;;
            q)
                exit 0
                ;;
            *)
                echo "Error: Please enter 'a', 'c', 'f', 'o', or 'q'" >&2
                ;;
        esac
    done
}

move_completed_todo() {
    local todo_text="$1"
    local category="$2"
    # Escape special characters in todo_text for use in sed
    local escaped_todo
    escaped_todo=$(printf '%s\n' "$todo_text" | sed 's/[[\.*^$/]/\\&/g')
    # Find the line number of the ### Category section under [x.x.x]
    local pending_section_line
    local category_line
    local actual_category_line
    pending_section_line=$(grep -n "^## \[x\.x\.x\]" "$CHANGELOG_PATH" | cut -d: -f1)
    category_line=$(sed -n "$((pending_section_line + 1)),\$p" "$CHANGELOG_PATH" | grep -n "^### $category" | head -1 | cut -d: -f1)
    actual_category_line=$((pending_section_line + category_line))
    # Check if there are already items in this category section
    # Look for the first line starting with '- ' after the category header
    local next_content_line
    local next_line_content
    next_content_line=$((actual_category_line + 1))
    next_line_content=$(sed -n "${next_content_line}p" "$CHANGELOG_PATH")
    # Remove the TODO from the TODO section
    sed -i "/^- $escaped_todo$/d" "$CHANGELOG_PATH"
    # Determine insertion point and format
    if [[ "$next_line_content" =~ ^-\  ]]; then
        # There's already an item, insert right after category header with blank line first
        sed -i "${actual_category_line}a\\
\\
- $todo_text" "$CHANGELOG_PATH"
    elif [[ -z "$next_line_content" ]] || [[ "$next_line_content" =~ ^$ ]]; then
        # Next line is blank or empty, check if there are items after the blank line
        local after_blank_line
        local after_blank_content
        after_blank_line=$((next_content_line + 1))
        after_blank_content=$(sed -n "${after_blank_line}p" "$CHANGELOG_PATH")
        if [[ "$after_blank_content" =~ ^-\  ]]; then
            # Items exist after blank line, find the last item in this section
            local last_item_line
            last_item_line=$(awk -v start="$after_blank_line" '
                NR >= start && /^- / { last = NR }
                NR >= start && /^(###|##)/ { exit }
                END { print last }
            ' "$CHANGELOG_PATH")
            # Insert after the last item
            sed -i "${last_item_line}a\\
- $todo_text" "$CHANGELOG_PATH"
        else
            # This is the first item, add with blank line
            sed -i "${actual_category_line}a\\
\\
- $todo_text" "$CHANGELOG_PATH"
        fi
    else
        # Next line is not blank and not an item, this is first item
        sed -i "${actual_category_line}a\\
\\
- $todo_text" "$CHANGELOG_PATH"
    fi
    echo "CHANGELOG.md updated"
}

# Main
mapfile -t todos < <(get_todos)
printf '%s\n' "${todos[@]}"

SELECTED_TODO_NUM=""
SELECTED_TODO_TEXT=""
SELECTED_CATEGORY=""

# Loop to allow selecting a different TODO with 'o' option
while true; do
    select_todo "${todos[@]}"
    # Try to categorize the selected TODO
    if categorize_todo "$SELECTED_TODO_NUM" "$SELECTED_TODO_TEXT"; then
        # Categorization successful (return code 0), proceed to move
        break
    else
        # Categorization returned 1 (user selected 'o' for other TODO)
        # Re-display the TODO list and loop back
        echo
        printf '%s\n' "${todos[@]}"
    fi
done

move_completed_todo "$SELECTED_TODO_TEXT" "$SELECTED_CATEGORY"

# All features implemented:
# - Extract and display TODOs with numbering
# - Interactive TODO selection with validation
# - Category selection (Added/Changed/Fixed) with validation
# - 'o)ther TODO' option to reselect
# - 'q)uit' option to exit
# - Proper blank line handling in CHANGELOG.md
# - Move TODO from ## TODO to ## [x.x.x] sections
