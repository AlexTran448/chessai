import cv2
import numpy as np
import time
from sklearn.metrics import pairwise_distances
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from skimage.morphology import skeletonize
from skimage import measure

from chessai.common import *
from chessai.config import *
warp_height = 640
warp_width = 640
river_model = YOLO('river.pt')
model = YOLO('last.pt')


class BoardAligner:
    def __init__(
        self,
        ref_image_path,
        smooth=False,
        debug=False,
        output_video_path=None,
    ):
        self.force_update = False
        self.smooth = smooth
        self.debug = debug
        self.output_video_path = output_video_path

        # Transform matrices
        self.last_M_update = time.time()
        self.M = None
        self.M_inv = None
        self.h_array = []
        self.flip = False

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = 5
        parameters.errorCorrectionRate = 0.3
        self.parameters = parameters
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, parameters)

        # Load reference image
        self.ref_image = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)

        # Detect markers in reference image
        self.ref_corners, self.ref_ids, self.ref_rejected = self.detector.detectMarkers(
            self.ref_image
        )

        # Create bounding box from reference image dimensions
        self.rect = np.array(
            [
                [
                    [0, 0],
                    [self.ref_image.shape[1], 0],
                    [self.ref_image.shape[1], self.ref_image.shape[0]],
                    [0, self.ref_image.shape[0]],
                ]
            ],
            dtype="float32",
        )

        if self.output_video_path:
            self.output_video = cv2.VideoWriter(
                "output.avi",
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                10,
                (self.ref_image.shape[1], self.ref_image.shape[0]),
            )
        else:
            self.output_video = None

    def get_output_size(self):
        return self.ref_image.shape[1], self.ref_image.shape[0]

    def process(self, image, visualize=None):
        """Transform image"""
        if self.force_update:
            h = self.get_homography(image)
            print(h)
            if h is not None:
                self.M = h
                self.M_inv = np.linalg.inv(h)
                self.force_update = False
        if self.M is None:
            return False, image
        image = cv2.resize(image, (1280, 1280))
        cv2.imshow("original", image)
        src =[[0, 0], [0, 640], [640, 640], [640, 0]]
        src = np.array(src, dtype=np.float32)
        src = src.reshape(-1, 1, 2)
        dst = src * 2
        scale = cv2.findHomography(dst, src)[0]
        src = [[0, 0], [0, 640], [640, 640], [640, 0]]
        dst = [[10, 0], [10, 640], [650, 640], [650, 0]]
        src = np.array(src, dtype=np.float32)
        src = src.reshape(-1, 1, 2)
        dst = np.array(dst, dtype=np.float32)
        dst = dst.reshape(-1, 1, 2)
        shift = cv2.findHomography(src, dst)[0]
        true_M = np.matmul(self.M, scale)
        #true_M = np.matmul(true_M, shift)
        warped = cv2.warpPerspective(image, true_M, (18 * 60, 20 * 60))
        warped = cv2.resize(warped, (640, 640))
        cv2.imshow("warped", warped)
        cv2.waitKey(1)
        return True, warped

    def _process_aruco(self, gray, update_transform_matrices=True):
        """Find aruco and update transformation matrices"""

        # Detect aruco markers in gray frame
        res_corners, res_ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters
        )
        self.res_corners = res_corners
        self.res_ids = res_ids

        # If markers were not detected
        if res_ids is None:
            return False

        if not update_transform_matrices:
            return True

        # Find which markers in frame match those in reference image
        idx = which(self.ref_ids, res_ids)

        # If # of detected points is too small => ignore the result
        if len(idx) <= 2:
            return False

        # Flatten the array of corners in the frame and reference image
        these_res_corners = np.concatenate(res_corners, axis=1)
        these_ref_corners = np.concatenate([self.ref_corners[x] for x in idx], axis=1)

        # Estimate homography matrix
        try:
            h, s = cv2.findHomography(
                these_ref_corners, these_res_corners, cv2.RANSAC, 1000.0
            )
        except:
            return False

        # If we want smoothing
        if self.smooth:
            self.h_array.append(h)
            self.M = np.mean(self.h_array, axis=0)
        else:
            self.M = h

        self.M_inv, s = cv2.findHomography(
            these_res_corners, these_ref_corners, cv2.RANSAC, 1000.0
        )

        return True

    def get_homography(self, frame):
        cv2.waitKey(1)
        frame = cv2.resize(frame, (640, 640))
        results = model(frame, stream=True, classes=[0, 1])

        for result in results:
            piece_mask = np.zeros((640,640), dtype=np.uint8)
            piece_masks = []
            boxes = result.boxes.xyxy.cpu().numpy()
            piece = np.zeros((0, 2))
            masks = result.masks
            labels = result.boxes.cls
            piece_info = {}
            board_mask = np.zeros_like(piece_mask)
            regions = []

            if masks is not None:
                for i in range(len(masks)):
                    mask = masks[i]
                    label = int(labels[i].item())
                    if label == 0:
                        mask = mask.data[0].cpu().numpy()
                        board_mask_temp = (mask * 255).astype(np.uint8)
                        cv2.imshow("board_mask_temp", board_mask_temp)
                        # split board_mask into connected components
                        board_mask_temp = cv2.morphologyEx(board_mask_temp, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)), iterations=2)

                        output = measure.label(board_mask_temp)
                        regions = measure.regionprops(output)
                    if label == 1:
                        box = boxes[i]
                        x1, y1, x2, y2 = box
                        centre = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                        piece = np.append(piece, [centre], axis=0)
                        mask = cv2.resize(mask.data[0].cpu().numpy(), (640,640))
                        piece_mask += (mask * 255).astype(np.uint8)
                        piece_masks.append((mask * 255).astype(np.uint8))
                        piece_info[tuple(centre)] = (mask * 255).astype(np.uint8)
            max_region = None
            for region in regions:
                if max_region is None:
                    max_region = region
                if region.area > max_region.area:
                    max_region = region
            for region in regions:
                if region.area < max_region.area * 0.2:
                    board_mask[region.coords[:, 0], region.coords[:, 1]] = 0
                else:
                    board_mask[region.coords[:, 0], region.coords[:, 1]] = 255
            if board_mask is None:
                continue
            # get contours of the piece mask
            contours = cv2.findContours(board_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
            # draw contours
            # cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

            connected = []

            pieces_to_check = piece
            new_connection = True
            test_board = board_mask.copy()
            while new_connection:
                new_connection = False
                for p in pieces_to_check:
                    mask = piece_info[tuple(p)]
                    eclipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                    mask = cv2.dilate(mask, eclipse, iterations=3)
                    # Ensure the mask is the same size as test_board
                    mask = cv2.resize(mask, (test_board.shape[1], test_board.shape[0]))

                    # Now you can safely perform the bitwise_and operation
                    check = cv2.bitwise_and(test_board, mask)
                    # show check

                    if check.any():
                        connected.append(p)
                        new_connection = True
                        pieces_to_check = np.delete(pieces_to_check, np.where((pieces_to_check == p).all(axis=1)),
                                                    axis=0)
                        test_board = cv2.bitwise_or(test_board, mask)

            # copy frame then show board mask weighted
            # Ensure the bm is the same size as frame
            bm = cv2.cvtColor(board_mask, cv2.COLOR_GRAY2BGR)
            # Ensure the bm is the same size as frame
            bm = cv2.resize(bm, (frame.shape[1], frame.shape[0]))
            bm = cv2.addWeighted(frame, 0.5, bm, 0.5, 0)

            cv2.imshow("board mask", bm)

            board_mask = cv2.erode(board_mask, np.ones((5, 5), np.uint8), iterations=1)
            connected = np.array(connected)

            # get contours of the piece mask
            if contours is None:
                continue

            if connected is None:
                connected = np.zeros((0, 2))
            for contour in contours:
                contour = contour.reshape(-1, 2)
                if connected.shape[0] == 0:
                    connected = contour
                else:
                    connected = np.append(connected, contour, axis=0)

            # show connected points
            connected = connected.astype(np.int32)
            # cv2.drawContours(frame, [connected], -1, (0, 0, 255), 2)
            if connected.size > 0:
                hull = cv2.convexHull(connected.astype(np.int32))
            else: continue
            # draw hull
            #cv2.polylines(frame, [hull], True, (0, 255, 0), 2)
            # approximate hull as 4 points
            epsilon = 0.1 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            # draw approx
            # cv2.polylines(frame, [approx], True, (255, 0, 0), 2)
            # get the corners of the approx
            frame_copy = frame.copy()
            #cv2.drawContours(frame_copy, [approx], -1, (0, 255, 0), 2)
            #cv2.imshow("approx", frame_copy)
            if cv2.contourArea(approx) > 1000 and len(approx) == 4:
                h1, _ = cv2.findHomography(approx, np.array(
                    [[0, 0], [0, warp_height], [warp_width, warp_height], [warp_width, 0]]))

                inboard = cv2.warpPerspective(piece_mask, h1, (warp_width, warp_height))
                board = cv2.warpPerspective(frame, h1, (warp_width, warp_height))
                cv2.imshow("board", board)
                # get warped piece centers
                results = river_model.predict(board, stream=True, classes=[0], conf=0.1)
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xywh.cpu().numpy()
                        if len(boxes) == 0:
                                continue
                        else:
                            boxes = boxes[0]
                        if boxes[2] < boxes[3]:
                            # flip image 90 degrees
                            inboard = cv2.rotate(inboard, cv2.ROTATE_90_CLOCKWISE)
                            board = cv2.rotate(board, cv2.ROTATE_90_CLOCKWISE)
                            self.flip = True
                        else:
                            self.flip = False

                # check if the board is correct by checking if the 9 vertical and 10 horizontal lines matches the canny edge of the board
                # get black and white board image
                gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                canny = cv2.Canny(gray, 100, 100)
                cv2.imshow("canny", canny)
                inboard = cv2.bitwise_not(inboard)
                inboard = cv2.erode(inboard, np.ones((7, 7), np.uint8), iterations=2)
                canny = cv2.bitwise_and(canny, inboard)
                cv2.imshow("canny2", canny)


                canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
                canny = skeletonize(canny / 255).astype(np.uint8) * 255
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return None

                lines = cv2.HoughLines(canny, 2, np.pi / 180, 100)
                horizontal = []
                vertical = []

                if lines is not None and len(lines) > 2:
                    for line in lines:
                        line = line[0]
                        rho, theta = line
                        # remove lines that are not vertical or horizontal
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho

                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))

                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))

                        if theta > (11 * np.pi / 12) or theta < (np.pi / 12):
                            vertical.append(line)
                            cv2.line(board, (x1, y1), (x2, y2), (0, 0, 255, 0.5), 2)

                        if (np.pi * 11 / 24) < theta < (np.pi * 13 / 24):
                            horizontal.append(line)
                            cv2.line(board, (x1, y1), (x2, y2), (0, 0, 255, 0.5), 2)

                    cv2.imshow("lines", board)
                    # get x when y = 330
                    vertical_cross = []
                    for rho, theta in vertical:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        if abs(a) > 1e-6:  # Check if a is not too close to zero
                            x = int(x0 + (330 - y0) * (-b) / a)
                        else:
                            x = int(x0)  # If a is too close to zero, use x0 as the x-coordinate
                        vertical_cross.append(x)

                    if len(vertical_cross) < 2:
                        continue

                    dbscan = DBSCAN(eps=20, min_samples=1, metric='euclidean')
                    dbscan.fit(np.array(vertical_cross).reshape(-1, 1))
                    labels = dbscan.labels_
                    unique_labels = set(labels)

                    # subtract pi from theta if theta is greater than 5pi/6
                    for i in range(len(vertical)):
                        if vertical[i][1] > 5 * np.pi / 6:
                            vertical[i][1] -= np.pi
                    # convert all rho to positive
                    for i in range(len(vertical)):
                        if vertical[i][0] < 0:
                            vertical[i][0] = -vertical[i][0]

                    centroids = []
                    for label in unique_labels:
                        if label == -1:
                            continue
                        points = np.array(vertical)[labels == label]
                        centroids.append(np.mean(points, axis=0))

                    for centroid in centroids:
                        rho, theta = centroid
                        if theta < 0:
                            theta += np.pi
                            rho = -rho

                    centroids_images = board.copy()
                    vertical = centroids
                    for centroid in centroids:
                        rho, theta = centroid
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))

                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        cv2.line(centroids_images, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    horizontal_cross = []
                    for rho, theta in horizontal:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        y = int(y0 + (330 - x0) * a / b)
                        horizontal_cross.append(y)

                    if len(horizontal_cross) < 2:
                        continue
                    dbscan = DBSCAN(eps=20, min_samples=1, metric='euclidean')
                    dbscan.fit(np.array(horizontal_cross).reshape(-1, 1))
                    labels = dbscan.labels_
                    unique_labels = set(labels)

                    centroids = []
                    for label in unique_labels:
                        if label == -1:
                            continue
                        points = np.array(horizontal)[labels == label]
                        centroids.append(np.mean(points, axis=0))
                    horizontal = centroids
                    for centroid in centroids:
                        rho, theta = centroid
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))

                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))

                        cv2.line(centroids_images, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cv2.imshow("centroids", centroids_images)
                    # find gaps in centroid
                    # remove lines that are too close to the edge
                    print("v")
                    print(vertical)
                    print("h")
                    print(horizontal)
                    vertical = [v for v in vertical if 40 < v[0] < 600]
                    horizontal = [h for h in horizontal if 40 < h[0] < 600]
                    if len(vertical) < 2 or len(horizontal) < 2:
                        continue
                    vertical = sorted(vertical, key=lambda x: x[0])
                    horizontal = sorted(horizontal, key=lambda x: x[0])

                    # predicted squares between lines
                    v_min = vertical[0]
                    v_max = vertical[-1]

                    h_min = horizontal[0]
                    h_max = horizontal[-1]

                    # calculate intersection
                    intersections = []
                    for v in vertical:
                        for h in horizontal:
                            a = np.array([[np.cos(v[1]), np.sin(v[1])], [np.cos(h[1]), np.sin(h[1])]])
                            b = np.array([v[0], h[0]])
                            x = np.linalg.solve(a, b)
                            intersections.append(x)
                    intersections = np.array(intersections)
                    # print(intersections)
                    # calculate intersection of h_min and v_min

                    #
                    a = intersections[0]
                    b = intersections[len(horizontal) - 1]
                    c = intersections[-len(horizontal)]
                    d = intersections[-1]
                    homography_src = np.array([a, b, c, d])

                    matches = 0
                    best_i = 0
                    best_j = 0
                    saved_pred = [[0, 0]]
                    saved_h = None
                    # i matches with number of horizontal lines
                    # j matches with number of vertical lines
                    for i in range(min(4, len(vertical) - 1), 9):
                        for j in range(min(3, len(horizontal) - 1), 10):
                            # i and j are units of squares
                            # get homography of i and j

                            homography_dst = np.array([[0, 0], [0, j], [i, 0], [i, j]]).astype(np.float32)
                            h, _ = cv2.findHomography(homography_dst, homography_src)
                            # get predicted intersections which are a grid of i x j

                            pred = np.array([[x, y] for x in range(i + 1) for y in range(j + 1)])
                            if pred is None:
                                continue
                            pred = pred.astype(np.float32)
                            pred = cv2.perspectiveTransform(pred.reshape(-1, 1, 2), h).reshape(-1, 2)
                            # there are missing intersections but not missing predictions
                            # count intersections that are close to predictions within 10 pixels
                            distance = pairwise_distances(intersections, pred)
                            if sum(np.min(distance, axis=0) < 10) > matches:
                                saved_h = h
                                matches = sum(np.min(distance, axis=0) < 10)
                                saved_pred = pred

                    # predict all intersections
                    # draw predicted intersections

                    top = np.min(saved_pred[:, 1])
                    bottom = np.max(saved_pred[:, 1])
                    left = np.min(saved_pred[:, 0])
                    right = np.max(saved_pred[:, 0])

                    # draw pred
                    for p in saved_pred:
                        cv2.circle(board, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)

                    ratio = warp_height / 9
                    top = round(top / ratio) * 2 + 1
                    bottom = round(bottom / ratio) * 2 + 1
                    ratio = warp_width / 8
                    left = round(left / ratio) * 2 + 1
                    right = round(right / ratio) * 2 + 1
                    print(top, bottom, left, right)

                    pred = [[x, y] for x in range(left * 60, right * 60 + 1, 2 * 60) for y in
                            range(top * 60, bottom * 60 + 1, 2 * 60)]
                    print(pred)
                    if pred is None or saved_h is None:
                        continue
                    pred = np.array(pred).astype(np.float32)
                    #print(saved_pred)
                    print(len(pred), len(saved_pred))
                    if len(pred) == len(saved_pred):
                        h2, _ = cv2.findHomography(saved_pred, pred)
                        if self.flip:
                            # flip image 90 degrees
                            src = [[0, 0], [0, 640], [640, 640], [640, 0]]
                            dst = [[0, 640], [640, 640], [640, 0], [0, 0]]
                            src = np.array(src, dtype=np.float32)
                            src = src.reshape(-1, 1, 2)
                            dst = np.array(dst, dtype=np.float32)
                            dst = dst.reshape(-1, 1, 2)
                            flip_homo, _ = cv2.findHomography(dst, src)
                            h1 = np.matmul(flip_homo, h1)


                        h = np.matmul(h2, h1)
                        return h
        return None


    def update_homography(self):
        self.force_update = True
        print("updating")