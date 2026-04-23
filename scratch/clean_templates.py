import os
import json
import shutil
import glob

def clean_old_templates():
    src_dir = "templates"
    backup_dir = "templates_backup"
    
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    all_files = glob.glob(f"{src_dir}/**/*.json", recursive=True)
    moved_count = 0
    kept_count = 0

    print(f"Scanning {len(all_files)} templates for Gen-2.5 compatibility...")

    for filepath in all_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Check if this is a Nose-Relative (True Gen-2.5) file
            is_new = False
            if len(data) > 0:
                first_frame = data[0]
                if "face" in first_frame and first_frame["face"] is not None:
                    if "hands" in first_frame and len(first_frame["hands"]) > 0:
                        wrist_x = first_frame["hands"][0]["wrist_pos"]["x"]
                        # If x > 0.3, it means it's an absolute screen coordinate (0..1)
                        # Nose-relative should be close to 0 (-0.2 to 0.2 usually)
                        if abs(wrist_x) < 0.3:
                            is_new = True
            
            if not is_new:
                # Move to backup
                rel_path = os.path.relpath(filepath, src_dir)
                dest_path = os.path.join(backup_dir, rel_path)
                
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(filepath, dest_path)
                moved_count += 1
                print(f"[MOVED -> Backup] {rel_path}")
            else:
                kept_count += 1
                print(f"[KEPT -> Valid] {os.path.relpath(filepath, src_dir)}")

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    print(f"\n--- Cleanup Complete ---")
    print(f"Moved {moved_count} old templates to '{backup_dir}'")
    print(f"Kept {kept_count} Gen-2.5 templates in '{src_dir}'")

if __name__ == "__main__":
    clean_old_templates()
