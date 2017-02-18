from captcha.image import ImageCaptcha
from lstm_ocr import gen_rand
import cv2
import numpy as np
import glob

# captcha = ImageCaptcha(fonts=['/Users/lonica/Downloads/Xerox.ttf'], height=30, width=80, font_sizes=(32,33,34))
captcha = ImageCaptcha(fonts=['/Users/lonica/Downloads/Xerox.ttf'])


def gen_captcha(i):
    num = gen_rand()
    img = captcha.generate(num)

    img = np.fromstring(img.getvalue(), dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (80, 30))
    filename = 'samples/%d_%s.png' % (i, num)
    cv2.imwrite(filename, img)
    return num
    # img = img.transpose(1, 0)
    # img = img.reshape((80 * 30))
    # img = np.multiply(img, 1/255.0)
    # print(img)


# ~/Documents/dev/workspace/mxnet/bin/im2rec ~/Documents/dev/workspace/mxnet/example/warpctc/samples/image.lst ~/Documents/dev/workspace/mxnet/example/warpctc/ ~/Documents/dev/workspace/mxnet/example/warpctc/samples/train.rec

if __name__ == "__main__":
    with open('samples/image.lst', 'w') as f:
        for i in range(320000):
            num = gen_captcha(i + 1)
            f.write("%d\t%s\t%s\n" % (i+1, num, 'samples/%d_%s.png' % (i+1, num)))
        f.flush()
        f.close()
    # captcha.write('3782', 'samples/out.png')
    # f = glob.glob("samples/*.png")
    # print(f)
    # img = cv2.imread(f[0], cv2.IMREAD_GRAYSCALE)
    # print img.shape
    # img = img.transpose(1, 0)
    # img = img.reshape((80 * 30))
    # img = np.multiply(img, 1 / 255.0)
    # print(img)
    # import cv2.cv as cv
    # im = cv.LoadImage('samples/out.png')
    # thumb = cv.CreateImage((im.width / 2, im.height / 2), 8, 3)
    # cv.Resize(im, thumb)
    # cv.SaveImage("samples/thumb.png", thumb)
    # cv2.resize(img, (80, 30))
    print("success")
