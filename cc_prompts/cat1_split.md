
## **🤖 AI Coder Implementation Prompt - Updated**

**Task**: Split Category 1 BDDL files into Pick and Place atomic skills

**Context**: You have access to the `clean_bddl_final.py` script that successfully cleaned Category 2 BDDL files. Now you need to process Category 1 files which are compound "pick and place" skills.

**Input**: BDDL files in `original_44_skills/` folder that contain task names like:
- `put_the_black_bowl_on_the_plate`
- `stack_the_middle_black_bowl_on_the_back_black_bowl`
- `put_the_wine_bottle_in_the_cabinet`

**Output**: For each input file, generate 2 new BDDL files:
1. `{original_name}_pick.bddl` - Language: "pick {object_A}"
2. `{original_name}_place.bddl` - Language: "place {object_A} on/in/at/to {object_B}"

**Core Requirements**:

1. **Language Modification**:
   - **Pick files**: Change `(:language ...)` to `(:language pick {object_A})`
   - **Place files**: Change `(:language ...)` to `(:language place {object_A} on/in/at/to {object_B})`
   - Extract object_A and object_B from the original task name
   - Example 1: `put_the_black_bowl_on_the_plate` becomes:
     - Pick: `(:language pick black_bowl)`
     - Place: `(:language place black_bowl on plate)`
   - Example 2: `put the chocolate pudding to the left of the plate` becomes:
     - Pick: `(:language pick the chocolate pudding)`
     - Place: `(:language place the chocolate pudding to the left of the plate)`

2. **Object Filtering**:
   - **Pick files**: Keep only object_A + fixtures, clear all other objects
   - **Place files**: Keep object_A + object_B + fixtures, clear all other objects
   - **IMPORTANT**: Always keep objects that are targets of regions (e.g., cabinet sides, regions that target fixtures)

3. **Goal States**:
   - **Pick files**: Replace original goal with `(PickedUp {object_A})`
   - **Place files**: Keep original goal unchanged

4. **Init States**: Keep original init states for both files

5. **Regions**: Keep only regions that target kept objects or are used in kept init conditions

**Output Files**:
- Generate a JSON mapping file: `cat1_split_map.json`
- Key: original file absolute path
- Value: list of newly generated files absolute paths (pick + place)

**File Naming Convention**: 
- Input: `KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl`
- Output: 
  - `KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_pick.bddl`
  - `KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_place.bddl`

**Implementation Approach**:
- Use the existing `clean_bddl_final.py` as a base
- Modify the object filtering logic to handle the pick/place split
- Add logic to identify object_A and object_B from task names
- Implement the goal state modification for pick skills
- **CRITICAL**: Parse and modify the `(:language ...)` section for each file
- Generate two output files per input file
- Create a separate script (e.g., `split_cat1_bddl.py`) to handle this splitting logic

**Critical Notes**:
- Remember to keep objects that are targets of regions (like cabinet sides, regions targeting fixtures)
- **Language parsing is crucial**: You must extract object names from task names and generate appropriate language descriptions
- Test with a few files first to verify both the object filtering logic and language modification work correctly
