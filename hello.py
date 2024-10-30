from pathlib import Path
import cv2

def preprocess(path, kultivar, i):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    h,w = img.shape[:2]
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h,w = img.shape[:2]

    target_width = 224
    target_height = 336
    
    # Resize the image
    resized_image = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    path = "./dataset/{}_preprocessed/{}.png".format(kultivar, str(i))
    cv2.imwrite(path, resized_image)


def main():
    print("Hello from syzygium-samarangense-classification!")
    path_citra = Path("./dataset/citra")
    path_mdh = Path("./dataset/madu_deli_hijau")

    print("Preprocessing Jambu Citra Image...")
    i = 1
    for files in path_citra.iterdir():
        print(preprocess(files, "citra", i))
        i+=1
    
    print("Preprocessing Jambu MDH Image...")
    i = 1
    for files in path_mdh.iterdir():
        print(preprocess(files, "madu_deli_hijau", i))
        i+=1


if __name__ == "__main__":
    main()
