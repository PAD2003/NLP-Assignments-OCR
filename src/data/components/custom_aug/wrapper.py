from src.data.components.custom_aug.base import *
from src.data.components.custom_aug.augment import *
from src.data.components.custom_aug.draw import *

random.seed(None)

class Augmenter:
    translator = dict()
    texture = None
    def __init__(
        _self_, 
        dict_path="src/data/components/custom_aug/asset/translate.txt", 
        texture_path="src/data/components/custom_aug/asset/texture.png"
    ):
        if len(Augmenter.translator) == 0:
            with open(dict_path, "r") as file:
                for line in file:
                    part = line.strip().split()
                    if len(part) > 1:
                        Augmenter.translator[part[0]] = part[1]
        
        if Augmenter.texture is None:
            img = cv2.imread(texture_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("Cant read image")
            bg_color = np.mean(img, axis=(0, 1))
            threshold = (img < bg_color).astype(np.uint8)
            cropped = crop_img(img / 255, threshold, 0, 0, 0, 0, bg_color=1)
            cropped = cropped ** 2
            Augmenter.texture = cropped.copy()
    
    @staticmethod
    def translate(label):
        output = ""
        num_punc = 0
        for char in label:
            output += Augmenter.translator[char]
        
        for char in output:
            if char == '^' or char == '~' or char == '`' or char == '\'' or char == '?' or char == '.' or char == '_':
                num_punc += 1
        
        return output, len(output) - num_punc, num_punc

    @staticmethod
    def randomRange(range):
        if len(range) == 1:
            return range[0]
        else:
            return random.random() * (range[1] - range[0]) + range[0]
    @staticmethod
    def randomPick(select, p):
        return np.random.choice(select, p)

    @staticmethod
    def transform_img(img,  rotate=(-40, 40), 
                            shear_x=(-0.3, 0.3), 
                            shear_y=(-0.2, 0.2), 
                            logger=None,
                            debug= None,
                            p= 1,
                            keep_mask=False):
        return augment_img(img, rotate=rotate,
                                shear_x=shear_x,
                                shear_y=shear_y,
                                logger=logger,
                                debug=debug,
                                keep_mask=keep_mask)
    
    @staticmethod
    def add_noise(img, label,
                        mask=None,
                        bg_color=1,
                        line_width=(1, 4),
                        spacing=(1, 6),
                        y_noise=(0, 5),
                        skew_noise=(0, 0.1),
                        intent=(0.7, 0.3),
                        align=(-1.2, 1.2),
                        ignore_skew=(0.4, 0.6),
                        noise_level=(0, 0.1),
                        axis = (0.2, 0.8)):
        line_width = int(Augmenter.randomRange(line_width))
        y_noise = Augmenter.randomRange(y_noise)
        skew_noise = Augmenter.randomRange(skew_noise)
        intent = np.random.choice([False, True], p=intent)
        align = Augmenter.randomRange(align)
        ignore_skew = np.random.choice([False, True], p=ignore_skew)
        noise_level = Augmenter.randomRange(noise_level)
        axis = np.random.choice([0, 1], p=axis)
        spacing = max(Augmenter.randomRange(spacing), - line_width + 1)

        cnts, boxes, is_char, centroid = find_text_and_punc(img, mask, label)
        a, b = line_vector(cnts, 
                           boxes, 
                           is_char, 
                           centroid, 
                           img.shape, 
                           y_noise, 
                           skew_noise, 
                           intent=intent, 
                           align=align, 
                           ignore_skew=ignore_skew, 
                           axis=axis)
        
        # print(line_width)
        pattern = cv2.resize(Augmenter.texture, (line_width, line_width), interpolation=cv2.INTER_LINEAR)
        
        output = draw_line(img, pattern, spacing, a, b, axis=axis, noise_level=noise_level)
        return output

    @staticmethod
    def full_augment(img, label, choice=(0.05, 0.1, 0.1, 0.75)): # do nothing -- aug -- add line -- aug & add line
        p = np.random.choice((0, 1, 2, 3), p=choice)
        if p == 0:
            return img
        elif p == 1:
            return Augmenter.transform_img(img)
        elif p == 2:
            translated, _, _ = Augmenter.translate(label)
            # print(translated)
            return Augmenter.add_noise(img, translated, mask=None)
        elif p == 3:
            translated, _, _ = Augmenter.translate(label)
            # print(translated)
            transformed, transformed_mask = Augmenter.transform_img(img, keep_mask=True)
            cv2.imwrite("potato/output/mask.jpg", np.stack([transformed_mask, transformed_mask, transformed_mask], axis=2) * 255)
            output = Augmenter.add_noise(transformed, translated, mask=transformed_mask)
            return output
        else:
            return img

    @staticmethod
    def __call__(img, label, sample=1, p=None):
        output = []
        for i in range(sample):
            if p is None:
                output.append(Augmenter.full_augment(img, label))
            else:
                output.append(Augmenter.full_augment(img, label, p))

        return output 

if __name__ == "__main__":
    augmenter = Augmenter("./potato/asset/translate.txt", "./potato/asset/texture.png")
    img = cv2.imread("/work/hpc/firedogs/data_/new_train/train_img_{0}.{1}".format(2445, "jpg"))
    print(img.shape)
    output = augmenter(img, "gluco", 1)
    for i in range(len(output)):
        cv2.imwrite("/work/hpc/firedogs/potato/output/img_{}.jpg".format(i), output[i])
        
    # curr: aug_img = augmenter.process(img, "gluco", 1)[0]
    # new:  aug_img = augmenter.process(img, "gluco", 1, "train_img_2445")[0]
        
        
        