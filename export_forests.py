import joblib
import json
import os
import gc
import numpy as np

def export_forest_to_json(model_path, classes_path, output_json_path):
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return False
    if not os.path.exists(classes_path):
        print(f"Error: {classes_path} not found!")
        return False

    print(f"Loading {model_path}...")
    forest = joblib.load(model_path)
    classes = joblib.load(classes_path)

    classes_list = [str(c) for c in classes]
    n_features = int(forest.n_features_in_)
    n_classes = len(classes)
    n_estimators = len(forest.estimators_)

    print(f"Exporting {n_estimators} trees incrementally using SPARSE leaf representations...")
    
    # Open the file and write the JSON header
    with open(output_json_path, 'w', encoding='utf-8') as f:
        # Write classes, n_features, n_classes, and the start of the trees list
        f.write('{"classes":' + json.dumps(classes_list, separators=(',', ':')) + ',')
        f.write('"n_features":' + str(n_features) + ',')
        f.write('"n_classes":' + str(n_classes) + ',')
        f.write('"trees":[')

        # Serialize and write each tree one by one
        for idx, estimator in enumerate(forest.estimators_):
            tree = estimator.tree_
            
            # Normalize leaf values to probabilities and save only non-zero values (sparse)
            raw_values = tree.value # shape (n_nodes, 1, n_classes)
            normalized_values = []
            for val in raw_values:
                dist = val[0]
                s = dist.sum()
                if s > 0:
                    prob_dist = dist / s
                else:
                    prob_dist = dist
                
                # Build sparse dict: {class_idx_string: rounded_prob}
                sparse_dict = {}
                for c_idx, prob in enumerate(prob_dist):
                    if prob > 0.0001:
                        sparse_dict[str(c_idx)] = round(float(prob), 4)
                normalized_values.append(sparse_dict)

            # Round thresholds to 4 decimal places to minimize JSON size
            thresholds = [round(t, 4) if isinstance(t, float) else t for t in tree.threshold.tolist()]

            tree_data = {
                "children_left": tree.children_left.tolist(),
                "children_right": tree.children_right.tolist(),
                "feature": tree.feature.tolist(),
                "threshold": thresholds,
                "values": normalized_values
            }
            
            # Convert single tree to compact JSON
            tree_str = json.dumps(tree_data, separators=(',', ':'))
            f.write(tree_str)
            
            # Add comma between trees
            if idx < n_estimators - 1:
                f.write(',')
            
            # Free memory immediately for this tree
            del tree_data
            del thresholds
            del normalized_values
            if idx % 10 == 0:
                gc.collect()

        # Write the JSON footer
        f.write(']}')

    print(f"Successfully exported forest incrementally to {output_json_path}!")
    
    # Free memory
    del forest
    del classes
    gc.collect()
    return True

def main():
    if not os.path.exists("model_output"):
        os.makedirs("model_output")
        
    # Export Dynamic Model
    export_forest_to_json(
        "dynamic_ml_model.pkl",
        "dynamic_ml_classes.pkl",
        "model_output/dynamic_forest.json"
    )

    # Export Static Model
    export_forest_to_json(
        "static_ml_model.pkl",
        "static_ml_classes.pkl",
        "model_output/static_forest.json"
    )

if __name__ == "__main__":
    main()
