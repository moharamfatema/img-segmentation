# Image Segmentation using clustering methods

## Table of Contents

- [Image Segmentation using clustering methods](#image-segmentation-using-clustering-methods)
  - [Table of Contents](#table-of-contents)
  - [Exploring the dataset](#exploring-the-dataset)
  - [Exploring Data](#exploring-data)
  - [Display the images and their ground truth](#display-the-images-and-their-ground-truth)
  - [Segmentation using K-Means](#segmentation-using-k-means)
    - [Different number of clusters](#different-number-of-clusters)
    - [K-means Evaluation](#k-means-evaluation)

## Exploring the dataset

## Exploring Data

We use the [BSR_BSDS500][data] dataset, A large dataset of natural images that have been manually segmented. The human annotations serve as ground truth for learning grouping cues as well as a benchmark for comparing different segmentation and boundary detection algorithms. We report our results on the first 50 images of the test set.

Folder tree

```text
IN
|_Images
| |_val
| | |_ ....jpg
| | |_ Thumbs.db
| |_train
| | |_ ....jpg
| | |_ Thumbs.db
| |_test
|   |_ ....jpg
|   |_ Thumbs.db
|
|_groundTruth
  |_val
  | |_ ....mat
  |_train
  | |_ ....mat
  |_test
    |_ ....mat
```

Image shape: (321, 481, 3)

![explore][explore]

test samples:  50 50

## Display the images and their ground truth

We write the following function to display an image with its corresponding ground truth. We use the `imshow` function from `matplotlib` to display the images. We reverse the order of the color channels to display the images in RGB format, as opposed to the BGR format used by OpenCV.

```python
def disp_img_truth(img,truth,edges=True,gray=False):
  fig = plt.figure(figsize=(30,10))
  n = len(truth['groundTruth'][0])
  ax_img = fig.add_subplot(1,n+1,1)
  ax_img.imshow(img[:,:,::-1]) if not gray else ax_img.imshow(img)
  ax_img.axis('off')

  ax_gt = []
  for i,gt in enumerate(truth['groundTruth'][0]):
    ax_gt.append(fig.add_subplot(1,n+1,i+2))
    if edges:
      ground_truth = gt[0][0][1]
      ax_gt[-1].imshow(-1*ground_truth.astype(np.uint8),cmap='gray')
    else:
      ground_truth = gt[0][0][0]
      ax_gt[-1].imshow(ground_truth.astype(np.uint8))
    ax_gt[-1].axis('off')

  plt.show()
```

The function can show either the edges or the segments of the ground truth based on the input parameter `edges`.

```python
disp_img_truth(x_test[0],y_test[0],edges=False)
```

![segments][segments]

```python
disp_img_truth(x_test[0],y_test[0],edges=True)
```

![edges][edges]

## Segmentation using K-Means

We write a k-means class that is general enough to be used for any data.

```python
class K_Means:
  def __init__(self,k=3):
    self.k = k
    self.centroids = None
    self.labels = None

  def fit(self,x_train, max_iter=100):
    # put the data in the right format
    if x_train.ndim != 2:
      x_train = x_train.reshape((-1,x_train.shape[-1]))

    # initialize the centroids
    self.centroids = np.random.choice(x_train.shape[0],self.k,replace=False)
    self.centroids = x_train[self.centroids]

    it = 0
    while it < max_iter:
      it += 1
      # calculate the distance between each point and each centroid
      dist = np.linalg.norm(x_train[:,np.newaxis]-self.centroids[np.newaxis],axis=-1)
      # assign each point to the closest centroid
      self.labels = np.argmin(dist,axis=-1)
      # calculate the new centroids
      new_centroids = np.zeros_like(self.centroids)
      for i in range(self.k):
        # calculate the mean of all points assigned to the centroid
        new_centroids[i] = np.mean(x_train[self.labels==i],axis=0)
      # check the stopping condition
      if np.all(new_centroids == self.centroids):
        break
      self.centroids = new_centroids

  def predict(self,x_test):
    # put the data in the right format
    if x_test.ndim != 2:
      x_test = x_test.reshape((-1,x_test.shape[-1]))

    # calculate the distance between each point and each centroid
    dist = np.linalg.norm(x_test[:,np.newaxis]-self.centroids[np.newaxis],axis=-1)
    # assign each point to the closest centroid
    return np.argmin(dist,axis=-1)

```

The function `fit` takes as input the training data and the maximum number of iterations. It initializes the centroids randomly and then iterates until the centroids don't change or the maximum number of iterations is reached. The function `predict` takes as input the test data and returns the labels of the test data.

We then write a helper function to initialize the classifier, train it and predict the labels of the test data.

The `position` parameter will let us encode spatial information later usin a function `encode_position`. For now, we set the default to `False`.

```python
def k_means(k, img, position = False):
  fitted = K_Means(k)
  img = encode_position(img/255.) if position else (img / 255.)
  fitted.fit(img)
  labels = fitted.predict(img)

  return labels.reshape(img.shape[:2])
```

```python
plt.imshow(k_means(5,x_test[5]))
```

![kmeans][kmeans]

### Different number of clusters

For each image, we save the labels of the image for different numbers of clusters: `[3, 5, 7, 9, 11]`.

```python
def get_all_labels(X, n = [3,5,7,9,11], position = False):
  y_pred = []
  for x in X:
    y = []
    for k in n:
      y.append(k_means(k,x,position = position))
    y_pred.append(y)
  return y_pred
```

The following function will write the data as PNG images using openCV library.

```python
def save_all_labels(y_pred,dir):
  for i in range(len(y_pred)):
    path = f'{dir}/{i}'
    os.mkdir(path)
    for j,y in enumerate(y_pred[i]):
      y = 255. * y / max(np.max(y),1)
      cv2.imwrite(f'{path}/{j}.png',y)
```

```python
y_pred = get_all_labels(x_test)
```

```python
save_all_labels(y_pred, f'{OUT}/test')
```

View the saved result [here][test_label].

### K-means Evaluation

The following procedures will be carried on for each image I, each number of clusters K, and each ground truth M.

We evaluate our clustering usin external measures. We use the [F-measure][f] and the [Conditional Entropy][ce].

In order to compute either of the external measures correctly, we need to assign labels using maximum matching.

We calculate all contingency matrices. We use the function `sklearn.metrics.cluster.contingency_matrix` to calculate the contingency matrix.

```python
def contingency(y_pred,y_true):
    contingency_matrices = np.zeros((len(y_pred[0]),len(y_pred),),dtype=object)
    for k in range(len(y_pred[0])):
    # for each k in [3,5,7,9,11]
        for i in range(len(y_pred)):
            # for each image i
            temp = np.zeros((len(y_true[i]['groundTruth'][0]),),dtype = object)
            for m, gt in enumerate(y_true[i]['groundTruth'][0]):
                # each image i has m ground truths
                ground_truth = gt[0][0][0]
                temp[m] = contingency_matrix(ground_truth, y_pred[i][k])
            contingency_matrices[k,i] = temp.copy()
            del temp
    return contingency_matrices
```

<!-- References -->

[data]: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500
[test_label]:https://drive.google.com/drive/folders/10aHKCE2jgUvzB8fOljdiv-IE_ocjplM5?usp=share_link
[f]: https://en.wikipedia.org/wiki/F-score
[ce]: https://en.wikipedia.org/wiki/Conditional_entropy

[explore]: img/explore.png
[segments]: img/segments.png
[edges]: img/edges.png
[kmeans]: img/kmeans.png
