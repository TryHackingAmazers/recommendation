import os
import yolov_model as ym
from tqdm import tqdm
import numpy as np

items = ["Bed","Cabinet","Carpet","Ceramic floor","Chair","Closet","Cupboard","Curtains","Dining Table","Door","Frame","Futec frame","Futech tiles","Gypsum Board","Lamp","Nightstand","Shelf","Sideboard","Sofa","TV stand","Table","Transparent Closet","Wall Panel","Window","Wooden floor"]

map_items = {}
for i, item in enumerate(items):
    map_items[item] = i

def get_all_files_in_subfolders(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def process_training_images(folder, batch_size, model):
    image_paths = get_all_files_in_subfolders(folder)
    dataset = []

    # Split the list of image paths into batches
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch = image_paths[i:i + batch_size]
        results = model.predict(source=batch)

        # Process results for each image in the batch
        for result in results:
            objects = result.boxes.data.cpu().numpy()
            labels = result.names
            data = [0] * len(items)
            for obj in objects:
                data[map_items[labels[int(obj[5])]]] += 1
            dataset.append(data)
    dataset = np.array(dataset)
    mask = np.sum(dataset, axis=1) > 1
    dataset = dataset[mask]
    print(dataset.shape)
    np.save(open("checkpoint/dataset.npy", "wb"), dataset)

# process_training_images("/home/rohan/hackonama/recommendation/datasets/House_Room_Dataset",10,ym.load())
    
def generate_cova_matrix():
    if not os.path.isfile("checkpoint/dataset.npy"):
        process_training_images("/home/rohan/hackonama/datasets/House_Room_Dataset",10,ym.load())
    
    X = np.load("checkpoint/dataset.npy").T
    
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    X = (X - mean[:, np.newaxis]) / std[:, np.newaxis]
    corm = np.corrcoef(X)

    np.save(open("checkpoint/corm.npy", "wb"), corm)

if not os.path.isfile("./recommendation/checkpoint/corm.npy"):
    generate_cova_matrix()

corm = np.load("./recommendation/checkpoint/corm.npy")

def get_recommendations(input):
    def remove_common_elements(list1, list2):
        is_unique = lambda x: x not in list2
        unique_elements = list(filter(is_unique, list1))
        return unique_elements
    scores = input @ corm
    scores = scores.argsort()[::-1]
    suggestions = [items[i] for i in scores]
    orig = []
    for i in range(len(items)):
        if input[i] != 0:
            orig.append(items[i])
    suggestions = remove_common_elements(suggestions,orig)
    return suggestions

# score = ym.process_image("/home/rohan/hackonama/recommendation/TimberlandkingLSWENGE_0d80ca15-a0ad-4341-8b5e-4efa70f4c7a5.webp")
# print(score)
# reco = get_recommendations(score)


