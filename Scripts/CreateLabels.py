import os
import pickle

def createLabels(db):
    print("Starting to Create Labels for the Data....")
    label_ids = {}
    i_id = 0
    for root, dirs, files in os.walk(os.path.join(db, "Train")):
        for file in files:
            if file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                if not label in label_ids:
                    label_ids[label] = i_id
                    i_id += 1
                #print(label_ids[label]," - ",label)
    print(label_ids)
    with open("labels.pickle", "wb") as f:
        pickle.dump(label_ids, f) 
    print("Labeling of Data Completed.")
    return label_ids