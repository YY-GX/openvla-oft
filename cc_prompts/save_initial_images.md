# 🤖 AI Coder Implementation Prompt - Initial Image Capture for Atomic Skills

## **Task**: Create a script to capture initial scene images for all atomic skills

## **Context**: 
You have access to the LIBERO benchmark system with a newly created `atomic_skills` benchmark containing 76 tasks (12 atomic + 32 pick + 32 place skills). The user wants to see what each scene looks like without having to create proper initial state files.

## **Requirements**:

1. **Script Location**: Create the script in `scripts/atomic_skills_scripts/` directory
2. **Functionality**: Capture initial scene images for all 76 tasks in the `atomic_skills` benchmark
3. **Initial State Handling**: Since initial files don't exist yet, implement a method to random initialize scenes without setting specific initial states
4. **Output**: Save 76 images showing what each atomic skill scene looks like

## **Technical Approach**:

### **Scene Initialization Strategy**:
- **Random Initialization**: Instead of loading specific initial states, implement random object placement
- **Default Scene Setup**: Use the scene definitions from BDDL files but with randomized object positions
- **Fallback Method**: If random initialization fails, use default/neutral object positions
- **Scene Rendering**: Ensure the scene renders properly even without exact initial state files

### **Image Capture Implementation**:
- **Benchmark Loading**: Load the `atomic_skills` benchmark using `get_benchmark("atomic_skills")`
- **Task Iteration**: Loop through all 76 tasks using `get_num_tasks()` and `get_task(i)`
- **Scene Setup**: For each task, set up the scene with random/default object positions
- **Camera Configuration**: Set up appropriate camera angles to capture the full scene
- **Image Saving**: Save images with naming convention: `{task_name}_initial.png`
- **Output Directory**: Save all images to `initial_images/atomic_skills/`

### **Key Components to Use**:
- `libero.libero.benchmark` module for benchmark loading
- Scene rendering capabilities (likely robosuite-based)
- Image capture utilities
- Task loading and scene setup logic

## **Expected Output**:
- A Python script that successfully runs without errors
- 76 initial scene images saved to disk
- Clear naming convention for easy identification
- Images showing the basic scene layout for each atomic skill

## **Implementation Notes**:
- Focus on getting the scenes to render and capture images
- Don't worry about perfect initial state accuracy - just get something visible
- Handle any missing dependencies gracefully
- Provide clear error messages if specific tasks fail

## **After Implementation**:
1. **Explain your approach** for handling the missing initial state files
2. **Describe the logic** of your script implementation
3. **List any assumptions** you made about the scene setup
4. **Note any limitations** or areas that could be improved

## **File Structure**:
```
scripts/
└── atomic_skills_scripts/
    └── capture_initial_images.py
```

## **Success Criteria**:
- Script runs without crashing
- Generates 76 image files
- Each image shows a recognizable scene
- Clear documentation of approach and limitations

---

**Remember**: 
- The goal is to get a visual preview of all atomic skill scenes, not perfect accuracy. Focus on making it work with the current constraints.
- After implementation, you have permission to run by yourselves to ensure it's correct.
- Save initial images into imgs/atomic_skills_scene_images, and keep each image's file name same as the language part in the bddl files.
