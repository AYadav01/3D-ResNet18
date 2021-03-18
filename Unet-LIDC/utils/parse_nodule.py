import os
import ast
import numpy as np
import nrrd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw
import pickle
from skimage.measure import label
from skimage.color import gray2rgb
from skimage.color import label2rgb


def readXML(path_to_xml):
    infile = open(path_to_xml, 'r')
    soup = BeautifulSoup(infile, 'lxml')
    regions = soup.findAll('roi')
    masterDict = {}

    for i in range(len(regions)):
        if i <= len(regions):
            """
            findChild() - gives only the first child node
            findChildren() - gives all the child nodes
            findChilren().contents - gives the value of the child node
            """
            child = regions[i].findChildren()
            """
            get 'ImageZPosition' and 'imageSOP_UID' values
            """
            zPos = float(child[0].contents[0])
            sop = str(child[1].contents[0])
            """
            get 'x' and 'y' values of 'edgeMap' node
            """
            coords = []
            for arg in child:
                if len(arg.contents) > 1:
                    x_coords, y_coords = arg.contents[1], arg.contents[3]
                    coords.append((int(x_coords.contents[0]), int(y_coords.contents[0])))

            # Skip there are less than 10 contour points
            if len(coords) >= 10:
                if zPos not in masterDict:
                    masterDict[zPos] = [sop, coords]
                else:
                    coords_old = masterDict[zPos][1]
                    if len(coords) > len(coords_old):
                        masterDict[zPos] = [sop, coords]
                    else:
                        continue
            else:
                continue
    return masterDict

def get_zPosIndex(zPos, sop, header):
    header = ast.literal_eval(header["map"])
    for arg in header:
        if (zPos == float(header[arg][0])) and (sop == str(header[arg][1])):
            return arg

def save_volume(masterDict, image, name, save_mask_path, save_img_path):
    data, header = nrrd.read(image)
    zPos = sorted(list(masterDict.keys()))
    DontSave = False
    if len(zPos) == 0 or len(zPos) == 1:
        mask_3d, image_3d = np.zeros((512, 512)), np.zeros((512, 512))
    else:
        mask_3d, image_3d = np.zeros((len(zPos), 512, 512)), np.zeros((len(zPos), 512, 512))
    for index, arg in enumerate(zPos):
        idx = get_zPosIndex(arg, masterDict[arg][0], header)
        if idx:
            try:
                image_3d[index] = data[idx]
                coords = masterDict[arg][1]
                nodule_img = Image.new('L', (512, 512), 0)
                ImageDraw.Draw(nodule_img).polygon(coords, outline=1, fill=1)
                nodule = np.array(nodule_img)
                """
                # For visualizing nodule
                label_image = label(nodule)
                rgb_image = (data[idx]- data[idx].min()) / (data[idx].max() - data[idx].min())
                image_label_overlay = label2rgb(label_image, image=rgb_image, bg_label=0, kind='overlay')

                fig, axs = plt.subplots(nrows=1, ncols=3)
                axs[0].imshow(data[idx], cmap="gray")
                axs[1].imshow(nodule, cmap="gray")
                axs[2].imshow(image_label_overlay, cmap="gray")
                plt.show()
                """
                mask_3d[index] = nodule
            except Exception as e:
                print("nodule index is:", idx)
                print("Image {} cause exception".format(name))
        else:
            DontSave = True

    # Save image and mask
    if not DontSave:
        name = name.split(".nrrd")[0] + ".pickle"
        save_mask = os.path.join(save_mask_path, name)
        save_image = os.path.join(save_img_path, name)
        with open(save_mask, "bw") as file:
            pickle.dump(mask_3d, file)
        with open(save_image, "bw") as file:
            pickle.dump(image_3d, file)
    else:
        print("Saving skiped for file:", name)

def main(root_dir, save_mask_path, save_img_path):
    root_dict = {}
    written = 0
    folder_idx = 0
    for root, dirs, files in os.walk(root_dir):
        if len(files) == 2:
            print("Processing Folder: {}".format(folder_idx+1))
            xml_name, volume_name = files[0], files[1]
            xml_path = os.path.join(root, xml_name)
            volume = os.path.join(root, volume_name)
            try:
                masterDict = readXML(xml_path)
                save_volume(masterDict, volume, volume_name, save_mask_path, save_img_path)
            except Exception as e:
                print("Error reading xml file {}, saving skipped...".format(xml_name))
            folder_idx += 1
    print("All Files Read!")


if __name__ == "__main__":
    # root = "D:\\Processed_LIDC"
    root = "D:\\Processed_LIDC"
    save_mask_path = "D:\\Unet-3d\\mask_new"
    save_img_path = "D:\\Unet-3d\\image_new"
    main(root, save_mask_path, save_img_path)