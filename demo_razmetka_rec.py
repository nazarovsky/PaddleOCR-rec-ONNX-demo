# demo for ONNX text areas recognizer
import os, sys, glob, shutil
import numpy as np
import cv2
import onnxruntime
from rec_postprocess import CTCLabelDecode

CKPT_FOLDER = 'output/rec_db/'
CKPT_TEST_WEIGHTS_FILE = 'rec_db.onnx'
CHARACTER_DICT_PATH = 'output/rec_db/mm_dict.txt'
USE_SPACE_CHAR = False

    # define model
onnx_model_filename = os.path.join(CKPT_FOLDER,CKPT_TEST_WEIGHTS_FILE)
sess_rec = onnxruntime.InferenceSession(onnx_model_filename)

#deprocess image by mobilenet rules
def mobilenet_deprocess(in_img):
#    print('deprocess: {}'.format(in_img.shape))
    img = np.array(np.squeeze(in_img), np.float32)    
    # deprocess image to show by cv2 means
    img = np.true_divide(img, 2.0)
    img += 0.5
    img *= 255.0
    if len(img.shape) == 2: # if picture is gray - convert to cv2 BGR
       img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    img = img.astype(np.uint8)
    return img

#preprocess image by mobilenet rules
def mobilenet_preprocess(in_img):
#    print('preprocess: {}'.format(in_img.shape))
    img  = np.array(in_img, np.float32)
    # preprocess image to feed to neural network
    img /= 255.0 
    img -= 0.5
    img *= 2.0
    return img 

# draw a small text box on _img_ with _text_ at position x, y
def draw_text_box(img, text, x, y, fontColor = (255,255,255), backColor = (0,0,0), fontScale = 0.5, lineType = 1):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    # get the width and height of the text box
    t_w, t_h = cv2.getTextSize(text, font, fontScale=fontScale, thickness=lineType)[0]
    # make the coords of the box with a small padding of two pixels
    box_coords = [(int(x), int(y+5)), (int(x + t_w),int(y - t_h))]
    cv2.rectangle(img, box_coords[0], box_coords[1], backColor, cv2.FILLED)
    cv2.putText(img,'{}'.format(text), (int(x+1),int(y+1)), font, fontScale=fontScale, color=(0,0,0), thickness=lineType)
    cv2.putText(img,'{}'.format(text), (int(x),int(y)), font, fontScale=fontScale, color=fontColor, thickness=lineType)
    return


def recognize(session, image_src):

    print('ONNX input size is {}'.format(session.get_inputs()[0].shape))
    print('ONNX output size is {}'.format(session.get_outputs()[0].shape))
    if len(image_src.shape) == 2:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)

    IN_IMAGE_H, IN_IMAGE_W, IN_IMAGE_D = image_src.shape

    IN_ONNX_H  = session.get_inputs()[0].shape[2]
    IN_ONNX_W  = session.get_inputs()[0].shape[3]
    IN_PROP_H  = float(IN_ONNX_H/IN_IMAGE_H)
    IN_PROP_W  = float(IN_ONNX_W/IN_IMAGE_W)
#    # resize input
    resized = cv2.resize(image_src, (IN_ONNX_W, IN_ONNX_H), interpolation=cv2.INTER_LINEAR)
    img_in = mobilenet_preprocess(resized)
    print('img_in size is {} - {}'.format(img_in.shape, img_in.dtype))
    xb = np.array(np.expand_dims(np.transpose(img_in, (2, 0, 1)), axis = 0),np.float32)
#    print('xb size is {}'.format(xb.shape))
#    # save debug to txt
##    xb1 = np.squeeze(xb)
##    for k in range(3):
##        np.savetxt('image_meas{}.txt'.format(k), xb1[k,:,:] ,fmt='%+3.3f',delimiter=' ') 
#
    # run onnx inference session
    x = xb if isinstance(xb, list) else [xb]
    feed = dict([(input.name, x[n]) for n, input in enumerate(session.get_inputs())])
    pred_onnx   = session.run(None, feed)
#    print('pred_onnx output shape is {}'.format(pred_onnx[0].shape))
    # decode by CTC algorithm
    ctc = CTCLabelDecode(character_dict_path=CHARACTER_DICT_PATH, character_type='ch', use_space_char=USE_SPACE_CHAR)
    rec_text = ctc(pred_onnx[0])
    print('rec_text is is {}'.format(rec_text))
    added_image = image_src
    added_image = cv2.vconcat([added_image,added_image])
    draw_text_box(added_image, str(rec_text), 10, 15, fontColor = (0,255,0))
    cv2.imshow('added_image',added_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return added_image,rec_text

def main():
    for i, file_path in enumerate(glob.glob(os.path.join('output','rec*.jpg'))):
        frame =  cv2.imread(file_path)
        added_image, text_recognized = recognize(sess_rec, frame)
        fname =  os.path.splitext(file_path)[0]
        cv2.imwrite('{}_out.jpg'.format(fname), added_image)

if __name__ == '__main__':
    main()
