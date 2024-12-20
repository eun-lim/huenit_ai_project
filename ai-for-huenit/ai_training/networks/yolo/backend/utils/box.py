import numpy as np
import cv2

class BoundBox:
    def __init__(self, x, y, w, h, c = None, classes = None):
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        
        self.c     = c
        self.classes = classes

    def get_label(self):
        return np.argmax(self.classes)
    
    def get_score(self):
        return self.classes[self.get_label()]
    
    def iou(self, bound_box):
        b1 = self.as_centroid()
        b2 = bound_box.as_centroid()
        return centroid_box_iou(b1, b2)

    def as_centroid(self):
        return np.array([self.x, self.y, self.w, self.h])
    

def boxes_to_array(bound_boxes):
    """
    # Args
        boxes : list of BoundBox instances
    
    # Returns
        centroid_boxes : (N, 4)
        probs : (N, nb_classes)
    """
    centroid_boxes = []
    probs = []
    for box in bound_boxes:
        centroid_boxes.append([box.x, box.y, box.w, box.h])
        probs.append(box.classes)
    return np.array(centroid_boxes), np.array(probs)


def nms_boxes(boxes, n_classes, nms_threshold=0.3, obj_threshold=0.3):
    """
    # Args
        boxes : list of BoundBox
    
    # Returns
        boxes : list of BoundBox
            non maximum supressed BoundBox instances
    """
    # suppress non-maximal boxes
    for c in range(n_classes):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if boxes[index_i].iou(boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    return boxes


def draw_scaled_boxes(image, boxes, probs, labels, desired_size=400):
    img_size = min(image.shape[:2])
    if img_size < desired_size:
        scale_factor = float(desired_size) / img_size
    else:
        scale_factor = 1.0
    
    h, w = image.shape[:2]
    img_scaled = cv2.resize(image, (int(w*scale_factor), int(h*scale_factor)))
    if boxes != []:
        boxes_scaled = boxes*scale_factor
        boxes_scaled = boxes_scaled.astype(np.int32)
    else:
        boxes_scaled = boxes
    return draw_boxes(img_scaled, boxes_scaled, probs, labels)
        

def draw_boxes(image, boxes, scores, classes, labels):

    color = (0, 125, 0)

    for i in range(len(boxes)):

        x_min, y_min, x_max, y_max  = boxes[i]
        obj_class = classes[i]
        score = scores[i]

        # Draw bounding box around detected object
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        #print(labels[obj_class], score)
        # Create label for detected object class
        label = "{}:{:.2f}%".format(labels[obj_class], np.max(score))
        label_color = (255, 255, 255)

        text_size = 0.0015 * min(image.shape[0], image.shape[1])

        # Make sure label always stays on-screen
        x_text, y_text = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, text_size, 1)[0][:2]

        lbl_box_xy_min = (x_min, y_min if y_min < 25 else y_min - y_text)
        lbl_box_xy_max = (x_min + x_text, y_min + y_text if y_min < 25 else y_min)
        lbl_text_pos = (x_min, y_min)

        # Add label and confidence value
        cv2.rectangle(image, lbl_box_xy_min, lbl_box_xy_max, color, -1)
        cv2.putText(image, label, lbl_text_pos, cv2.FONT_HERSHEY_DUPLEX, text_size, label_color, 1, cv2.LINE_AA)

    return image        

def centroid_box_iou(box1, box2):
    def _interval_overlap(interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
    
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2,x4) - x3
    
    _, _, w1, h1 = box1.reshape(-1,)
    _, _, w2, h2 = box2.reshape(-1,)
    x1_min, y1_min, x1_max, y1_max = to_minmax(box1.reshape(-1,4)).reshape(-1,)
    x2_min, y2_min, x2_max, y2_max = to_minmax(box2.reshape(-1,4)).reshape(-1,)
            
    intersect_w = _interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = _interval_overlap([y1_min, y1_max], [y2_min, y2_max])
    intersect = intersect_w * intersect_h
    union = w1 * h1 + w2 * h2 - intersect
    
    return float(intersect) / union


def to_centroid(minmax_boxes):
    """
    minmax_boxes : (N, 4) [[100, 120, 140, 200]]
    centroid_boxes: [[120. 160.  40.  80.]]
    """
    #minmax_boxes = np.asarray([[100, 120, 140, 200]])
    minmax_boxes = minmax_boxes.astype(np.float64)
    centroid_boxes = np.zeros_like(minmax_boxes)
    
    x1 = minmax_boxes[:,0]
    y1 = minmax_boxes[:,1]
    x2 = minmax_boxes[:,2]
    y2 = minmax_boxes[:,3]
    
    centroid_boxes[:,0] = (x1 + x2) / 2
    centroid_boxes[:,1] = (y1 + y2) / 2
    centroid_boxes[:,2] = x2 - x1
    centroid_boxes[:,3] = y2 - y1
    return centroid_boxes

def to_minmax(centroid_boxes):
    centroid_boxes = centroid_boxes.astype(np.float64)
    minmax_boxes = np.zeros_like(centroid_boxes)
    
    cx = centroid_boxes[:,0]
    cy = centroid_boxes[:,1]
    w = centroid_boxes[:,2]
    h = centroid_boxes[:,3]
    
    minmax_boxes[:,0] = cx - w/2
    minmax_boxes[:,1] = cy - h/2
    minmax_boxes[:,2] = cx + w/2
    minmax_boxes[:,3] = cy + h/2
    return minmax_boxes

def create_anchor_boxes(anchors):
    """
    # Args
        anchors : list of floats
    # Returns
        boxes : array, shape of (len(anchors)/2, 4)
            centroid-type
    """
    boxes = []
    n_boxes = int(len(anchors)/2)
    for i in range(n_boxes):
        boxes.append(np.array([0, 0, anchors[2*i], anchors[2*i+1]]))
    return np.array(boxes)

def find_match_box(centroid_box, centroid_boxes):
    """Find the index of the boxes with the largest overlap among the N-boxes.

    # Args
        box : array, shape of (1, 4)
        boxes : array, shape of (N, 4)
    
    # Return
        match_index : int
    """
    match_index = -1
    max_iou     = -1
    
    for i, box in enumerate(centroid_boxes):
        iou = centroid_box_iou(centroid_box, box)
        
        if max_iou < iou:
            match_index = i
            max_iou     = iou
    return match_index

