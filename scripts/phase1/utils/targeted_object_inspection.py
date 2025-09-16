#!/usr/bin/env python3
"""
Create comprehensive MuJoCo object mapping by directly inspecting specific BDDL files.
"""

import sys
import os
import json
import glob
from pathlib import Path

# Add LIBERO paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

from libero.libero.envs import OffScreenRenderEnv


def inspect_specific_bddl(bddl_file, skill_name):
    """Inspect a specific BDDL file to extract object names."""
    
    print(f"\n🔍 Inspecting: {os.path.basename(bddl_file)}")
    print(f"For skill: {skill_name}")
    
    try:
        # Create environment
        env_args = {
            'bddl_file_name': bddl_file,
            'camera_heights': 256,
            'camera_widths': 256
        }
        
        env = OffScreenRenderEnv(**env_args)
        env.reset()
        
        # Get MuJoCo model
        model = env.sim.model
        
        # Collect all object names
        all_bodies = []
        all_geoms = []
        all_joints = []
        
        for i in range(model.nbody):
            body_name = model.body_id2name(i)
            if body_name:
                all_bodies.append(body_name)
        
        for i in range(model.ngeom):
            geom_name = model.geom_id2name(i)
            if geom_name:
                all_geoms.append(geom_name)
        
        for i in range(model.njnt):
            joint_name = model.joint_id2name(i)
            if joint_name:
                all_joints.append(joint_name)
        
        env.close()
        
        # Filter for object-related names (exclude robot/table/world)
        object_bodies = [b for b in all_bodies if not any(prefix in b for prefix in ['robot0', 'gripper0', 'mount0', 'world', 'table'])]
        object_geoms = [g for g in all_geoms if not any(prefix in g for prefix in ['robot0', 'gripper0', 'mount0', 'floor', 'wall', 'table'])]
        object_joints = [j for j in all_joints if not any(prefix in j for prefix in ['robot0', 'gripper0', 'mount0'])]
        
        print(f"  🎯 Scene objects found:")
        print(f"    Bodies: {object_bodies}")
        print(f"    Geoms: {object_geoms[:10]}{'...' if len(object_geoms) > 10 else ''}")
        print(f"    Joints: {object_joints}")
        
        return {
            'skill_name': skill_name,
            'bddl_file': os.path.basename(bddl_file),
            'object_bodies': object_bodies,
            'object_geoms': object_geoms,
            'object_joints': object_joints,
            'all_bodies': all_bodies,
            'all_geoms': all_geoms,
            'all_joints': all_joints
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {
            'skill_name': skill_name,
            'bddl_file': os.path.basename(bddl_file) if bddl_file else "None",
            'error': str(e)
        }


def main():
    """Create targeted object mapping."""
    
    # Define specific BDDL files for each skill type
    bddl_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills"
    
    skill_bddl_mappings = {
        # Moka pot skills
        "pick_moka_pot": "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove_pick.bddl",
        "place_moka_pot_on_stove": "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove_place.bddl",
        
        # Frying pan skills
        "pick_frying_pan": "modified_cat1/KITCHEN_SCENE3_put_the_frying_pan_on_the_stove_pick.bddl", 
        "place_frying_pan_on_stove": "modified_cat1/KITCHEN_SCENE3_put_the_frying_pan_on_the_stove_place.bddl",
        
        # Stove skills
        "turn_on_stove": "KITCHEN_SCENE3_turn_on_the_stove.bddl",
        "turn_off_stove": "KITCHEN_SCENE8_turn_off_the_stove.bddl",
        
        # Microwave skills  
        "open_microwave": "KITCHEN_SCENE7_open_the_microwave.bddl",
        "close_microwave": "KITCHEN_SCENE6_close_the_microwave.bddl",
        
        # Bowl skills
        "pick_black_bowl": "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_pick.bddl",
        "place_black_bowl_on_plate": "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_place.bddl",
        "pick_white_bowl": "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate_pick.bddl",
        "place_white_bowl_on_plate": "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate_place.bddl",
        
        # Drawer skills
        "open_top_drawer": "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet.bddl",
        "close_top_drawer": "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet.bddl",
        "open_bottom_drawer": "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet.bddl", 
        "close_bottom_drawer": "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet.bddl",
        
        # Ketchup skills
        "place_ketchup_in_drawer": "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet_place.bddl",
        
        # Wine bottle skills
        "pick_wine_bottle": "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack_pick.bddl",
        "place_wine_bottle_in_drawer": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_place.bddl"
    }
    
    print(f"🎯 Inspecting {len(skill_bddl_mappings)} specific BDDL files")
    
    inspection_results = []
    
    for skill_name, bddl_filename in skill_bddl_mappings.items():
        print("=" * 80)
        
        # Find the BDDL file
        bddl_file = os.path.join(bddl_dir, bddl_filename)
        if not os.path.exists(bddl_file):
            # Try in subdirectories
            search_patterns = [
                os.path.join(bddl_dir, "*", bddl_filename),
                os.path.join(bddl_dir, "**", bddl_filename)
            ]
            found = False
            for pattern in search_patterns:
                matches = glob.glob(pattern, recursive=True)
                if matches:
                    bddl_file = matches[0]
                    found = True
                    break
            
            if not found:
                print(f"⚠️  BDDL file not found: {bddl_filename}")
                inspection_results.append({
                    'skill_name': skill_name,
                    'bddl_file': bddl_filename,
                    'error': f'BDDL file not found: {bddl_filename}'
                })
                continue
        
        result = inspect_specific_bddl(bddl_file, skill_name)
        inspection_results.append(result)
    
    # Create comprehensive mapping
    object_name_mapping = {}
    
    for result in inspection_results:
        if 'error' not in result:
            skill_name = result['skill_name']
            
            # Extract primary objects for each skill type
            if 'moka_pot' in skill_name:
                moka_objects = [b for b in result['object_bodies'] if 'moka' in b.lower()]
                object_name_mapping['moka_pot'] = {
                    'main_body': moka_objects[0] if moka_objects else None,
                    'all_bodies': moka_objects,
                    'joints': [j for j in result['object_joints'] if 'moka' in j.lower()]
                }
                
            elif 'frying_pan' in skill_name:
                frying_objects = [b for b in result['object_bodies'] if 'frying' in b.lower() or 'frypan' in b.lower()]
                object_name_mapping['frying_pan'] = {
                    'main_body': frying_objects[0] if frying_objects else None,
                    'all_bodies': frying_objects,
                    'joints': [j for j in result['object_joints'] if 'frying' in j.lower() or 'frypan' in j.lower()]
                }
                
            elif 'stove' in skill_name:
                stove_objects = [b for b in result['object_bodies'] if 'stove' in b.lower()]
                object_name_mapping['stove'] = {
                    'main_body': [b for b in stove_objects if 'main' in b][0] if any('main' in b for b in stove_objects) else stove_objects[0] if stove_objects else None,
                    'burner_plate': [b for b in stove_objects if 'burner_plate' in b][0] if any('burner_plate' in b for b in stove_objects) else None,
                    'button': [b for b in stove_objects if 'button' in b][0] if any('button' in b for b in stove_objects) else None,
                    'all_bodies': stove_objects,
                    'joints': [j for j in result['object_joints'] if 'stove' in j.lower()]
                }
                
            elif 'microwave' in skill_name:
                microwave_objects = [b for b in result['object_bodies'] if 'microwave' in b.lower()]
                object_name_mapping['microwave'] = {
                    'main_body': [b for b in microwave_objects if 'main' in b][0] if any('main' in b for b in microwave_objects) else microwave_objects[0] if microwave_objects else None,
                    'door_body': [b for b in microwave_objects if 'door' in b][0] if any('door' in b for b in microwave_objects) else None,
                    'all_bodies': microwave_objects,
                    'joints': [j for j in result['object_joints'] if 'microwave' in j.lower()]
                }
                
            elif 'black_bowl' in skill_name:
                bowl_objects = [b for b in result['object_bodies'] if 'black' in b.lower() and 'bowl' in b.lower()]
                object_name_mapping['black_bowl'] = {
                    'main_body': bowl_objects[0] if bowl_objects else None,
                    'all_bodies': bowl_objects,
                    'joints': [j for j in result['object_joints'] if 'black' in j.lower() and 'bowl' in j.lower()]
                }
                
            elif 'white_bowl' in skill_name:
                bowl_objects = [b for b in result['object_bodies'] if 'white' in b.lower() and 'bowl' in b.lower()]
                object_name_mapping['white_bowl'] = {
                    'main_body': bowl_objects[0] if bowl_objects else None,
                    'all_bodies': bowl_objects,
                    'joints': [j for j in result['object_joints'] if 'white' in j.lower() and 'bowl' in j.lower()]
                }
                
            elif 'plate' in skill_name:
                plate_objects = [b for b in result['object_bodies'] if 'plate' in b.lower()]
                object_name_mapping['plate'] = {
                    'main_body': plate_objects[0] if plate_objects else None,
                    'all_bodies': plate_objects,
                    'joints': [j for j in result['object_joints'] if 'plate' in j.lower()]
                }
                
            elif 'drawer' in skill_name:
                cabinet_objects = [b for b in result['object_bodies'] if 'cabinet' in b.lower()]
                drawer_joints = [j for j in result['object_joints'] if 'level' in j.lower()]
                
                if 'top_drawer' not in object_name_mapping:
                    object_name_mapping['top_drawer'] = {
                        'main_body': [b for b in cabinet_objects if 'top' in b][0] if any('top' in b for b in cabinet_objects) else None,
                        'all_bodies': cabinet_objects,
                        'joint': [j for j in drawer_joints if 'top' in j][0] if any('top' in j for j in drawer_joints) else None
                    }
                    
                if 'bottom_drawer' not in object_name_mapping:
                    object_name_mapping['bottom_drawer'] = {
                        'main_body': [b for b in cabinet_objects if 'bottom' in b][0] if any('bottom' in b for b in cabinet_objects) else None,
                        'all_bodies': cabinet_objects, 
                        'joint': [j for j in drawer_joints if 'bottom' in j][0] if any('bottom' in j for j in drawer_joints) else None
                    }
                
            elif 'ketchup' in skill_name:
                ketchup_objects = [b for b in result['object_bodies'] if 'ketchup' in b.lower()]
                object_name_mapping['ketchup'] = {
                    'main_body': ketchup_objects[0] if ketchup_objects else None,
                    'all_bodies': ketchup_objects,
                    'joints': [j for j in result['object_joints'] if 'ketchup' in j.lower()]
                }
                
            elif 'wine_bottle' in skill_name:
                wine_objects = [b for b in result['object_bodies'] if 'wine' in b.lower()]
                object_name_mapping['wine_bottle'] = {
                    'main_body': wine_objects[0] if wine_objects else None,
                    'all_bodies': wine_objects,
                    'joints': [j for j in result['object_joints'] if 'wine' in j.lower()]
                }
    
    # Save comprehensive results
    output_file = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/prompts/from_ai/targeted_mujoco_objects.json"
    
    final_report = {
        'inspection_results': inspection_results,
        'object_name_mapping': object_name_mapping,
        'skill_specific_mappings': {
            # Map skills from skills_objects_information.txt to MuJoCo object names
            'pick moka pot': object_name_mapping.get('moka_pot', {}),
            'place moka pot on the stove 2': {
                'target_object': object_name_mapping.get('stove', {}),
                'manipulation_object': object_name_mapping.get('moka_pot', {})
            },
            'turn on the stove 2': object_name_mapping.get('stove', {}),
            'pick frying pan': object_name_mapping.get('frying_pan', {}),
            'place frying pan on the stove 1': {
                'target_object': object_name_mapping.get('stove', {}),
                'manipulation_object': object_name_mapping.get('frying_pan', {})
            },
            'turn on the stove 1': object_name_mapping.get('stove', {}),
            'open the microwave 1': object_name_mapping.get('microwave', {}),
            'pick black bowl': object_name_mapping.get('black_bowl', {}),
            'place black bowl on the plate 1': {
                'target_object': object_name_mapping.get('plate', {}),
                'manipulation_object': object_name_mapping.get('black_bowl', {})
            },
            'pick white bowl': object_name_mapping.get('white_bowl', {}),
            'place white bowl on the plate 2': {
                'target_object': object_name_mapping.get('plate', {}),
                'manipulation_object': object_name_mapping.get('white_bowl', {})
            },
            'open the top drawer of the cabinet 1': object_name_mapping.get('top_drawer', {}),
            'place ketchup in top drawer of the cabinet 1': {
                'target_object': object_name_mapping.get('top_drawer', {}),
                'manipulation_object': object_name_mapping.get('ketchup', {})
            },
            'close the top drawer of the cabinet 1': object_name_mapping.get('top_drawer', {}),
            'open the bottom drawer of the cabinet 1': object_name_mapping.get('bottom_drawer', {}),
            'place wine bottle in the bottom drawer of the cabinet 1': {
                'target_object': object_name_mapping.get('bottom_drawer', {}),
                'manipulation_object': object_name_mapping.get('wine_bottle', {})
            },
            'close the bottom drawer of the cabinet 1': object_name_mapping.get('bottom_drawer', {})
        },
        'usage_notes': {
            'pick_skills': 'Use manipulation_object.main_body for the object to grasp',
            'place_skills': 'Use target_object.main_body for placement target, manipulation_object.main_body for object being placed',
            'atomic_skills': 'Use main_body for the object to interact with, joint for articulated objects',
            'stove_interaction': 'Use button body for turn on/off, burner_plate for placing objects',
            'drawer_interaction': 'Use joint for opening/closing, main_body for placement target'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n📄 Targeted object mapping saved to: {output_file}")
    
    # Print summary
    print(f"\n📊 TARGETED MUJOCO OBJECT MAPPING:")
    print("=" * 80)
    
    for obj_type, mapping in object_name_mapping.items():
        print(f"\n🎯 {obj_type.upper()}:")
        for key, value in mapping.items():
            if key == 'all_bodies' and len(value) > 3:
                print(f"  {key}: {value[:3]}... ({len(value)} total)")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()