import cv2


class BBox:
    def __init__(self, x1, y1, x2, y2):  # , file_name, width, height, classname,
        # self.file_name = file_name
        # self.classname = classname
        # if x1 is None:
        #     return
        self._set_box_points(x1, y1, x2, y2)
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin
        self.flag = False

    def _set_box_points(self, x1, y1, x2, y2):
        if (x1 > x2):
            self.xmin = x2
            self.xmax = x1
        else:
            self.xmin = x1
            self.xmax = x2
        if (y1 > y2):
            self.ymax = y1
            self.ymin = y2
        else:
            self.ymax = y2
            self.ymin = y1

    def __str__(self):
        return "{},{},{},{},{},{},{},{}".format(self.file_name, self.width, self.height, self.classname,
                                                self.xmin, self.ymin, self.xmax, self.ymax)

    def get_IoU(self, bbox):
        box_a = [self.xmin, self.ymin, self.xmax, self.ymax]
        box_b = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
        
        # determine the (x, y)-coordinates of the intersection rectangle
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])
        # compute the area of intersection rectangle
        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = inter_area / float(box_a_area + box_b_area - inter_area)
        # return the intersection over union value
        return iou

    def get_IoO(self, bbox):
        box_a = [self.xmin, self.ymin, self.xmax, self.ymax]
        box_b = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]

        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])
        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
        box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
        # compute the intersection over object by taking the intersection
        # area and dividing it by the ground-truth
        ioo = inter_area / float(box_b_area)
        return ioo

    def paint(self, orig_image, color):
        cv2.rectangle(orig_image,
                      (int(self.xmin), int(self.ymin)),
                      (int(self.xmax), int(self.ymax)),
                      color, 2)
        return orig_image

