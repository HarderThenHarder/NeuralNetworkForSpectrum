import cv2
import numpy as np


class ColorMapping:

    def __init__(self, material_picture):
        
        # Color form -> (b, g, r)
        self.material_color_dict = {0: (255, 0, 0),         #bg
                                    1: (0, 255, 255),       #PP
                                    2: (0, 255, 0),         #PS
                                    3: (0, 0, 255),         #PE
                                    4: (255, 0, 168),       #RVS
                                    5: (0, 150, 255),       #PET
                                    6: (240, 255, 0),       #HDPE
                                    7: (242, 153, 252),     #PLA
                                    8: (150, 150, 150)}     #PVC
        
        self.material_name_dict = {0: 'Background',
                                   1: 'PP',
                                   2: 'PS',
                                   3: 'PE',
                                   4: 'RVS',
                                   5: 'PET',
                                   6: 'HDPE',
                                   7: 'PLA',
                                   8: 'PVC'}
        self.material_picture = material_picture
        self.width = self.material_picture.shape[1]
        self.height = self.material_picture.shape[0]
        self.result_picture = np.zeros((self.height, self.width, 3))

    def map_color(self):
        """
        # Transform the materail matrix to the color picture
        """
        for row in range(self.height):
            for col in range(self.width):
                self.result_picture[row][col] = self.material_color_dict[self.material_picture[row][col]]
        return self.result_picture

    def write_label(self):
        """
        # Write the color lable on the top-right of the result picture
        """
        idx = 0
        for key in self.material_name_dict:
            cv2.putText(self.result_picture, self.material_name_dict[key] + ": " + str(self.material_color_dict[key]), (int(0.7 * self.width),20 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.material_color_dict[key], 1)
            idx += 1

    def save_color_image(self, img_name: str) -> None:
        """
        # Save the color picture
        """
        cv2.imwrite(img_name, self.result_picture)


if __name__ == "__main__":
    material_picture = np.vstack([np.full((80, 520), i) for i in range(9)])
    cm = ColorMapping(material_picture)
    cm.map_color()
    cm.write_label()
    cm.save_color_image("result.jpg")
    print(material_picture)
