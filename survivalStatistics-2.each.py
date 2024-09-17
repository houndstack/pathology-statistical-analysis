import alphashape
import shapely
import torch
import numpy as np
import os
from sklearn.cluster import DBSCAN
from statistics import mode
from scipy.spatial.distance import directed_hausdorff
from sklearn.neighbors import KernelDensity
from scipy import stats
from scipy.spatial import distance
from scipy.stats import entropy, wasserstein_distance
import matplotlib.pyplot as plt
import ot
import sys

def stat(model, outPrefix):
  boxes = model['cell_stats']['boxes'].numpy()
  scores = model['cell_stats']['scores'].numpy()
  labels = model['cell_stats']['labels'].numpy()
  colors = model['meta_info']['labels_color']

  X = [((x[0]+x[2])/2) for i,x in enumerate(boxes) if labels[i]==1]
  y = [((x[1]+x[3])/2) for i,x in enumerate(boxes) if labels[i]==1]
  Xlympho = [((x[0]+x[2])/2) for i,x in enumerate(boxes) if labels[i]==3]
  ylympho = [((x[1]+x[3])/2) for i,x in enumerate(boxes) if labels[i]==3]


  featuresLympho = [[((x[0]+x[2])/2),((x[1]+x[3])/2)] for i,x in enumerate(boxes) if labels[i]==3]
  features = [[((x[0]+x[2])/2),((x[1]+x[3])/2)] for i,x in enumerate(boxes) if labels[i]==1]


  db = DBSCAN(eps=500, min_samples=10, algorithm='ball_tree', metric='euclidean').fit(features)
  label = db.labels_
  num_clusters = len(set(label))
  print(num_clusters)
  mostFrequentCluster = mode(label)
  tumorCluster = [x for i,x in enumerate(features) if label[i]==mostFrequentCluster]

  db = DBSCAN(eps=500, min_samples=10, algorithm='ball_tree', metric='euclidean').fit(featuresLympho)
  label = db.labels_
  num_clusters = len(set(label))
  print(num_clusters)
  mostFrequentCluster = mode(label)
  lymphoCluster = [x for i,x in enumerate(featuresLympho) if label[i]==mostFrequentCluster]


  DH = directed_hausdorff(tumorCluster, lymphoCluster)

  dist1 = np.array(tumorCluster)
  dist2 = np.array(lymphoCluster)

  hausdorff_dist = max(distance.directed_hausdorff(dist1, dist2)[0], distance.directed_hausdorff(dist2, dist1)[0])

  wasserstein = wasserstein_distance(dist1.ravel(), dist2.ravel())

  dist1_prob = np.histogram2d([row[0] for row in tumorCluster], [row[1] for row in tumorCluster], bins=20, density=True)[0].ravel()
  dist2_prob = np.histogram2d([row[0] for row in lymphoCluster], [row[1] for row in lymphoCluster], bins=20, density=True)[0].ravel()

  def jensen_shannon(p, q):
      p = np.asarray(p)
      q = np.asarray(q)
      m = 0.5 * (p + q)
      return 0.5 * (entropy(p, m) + entropy(q, m))

  js_divergence = jensen_shannon(dist1_prob, dist2_prob)

  kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(tumorCluster)
  log_density = kde.score_samples(tumorCluster)
  mean_log_density = np.mean(log_density)


  m1 = np.array([i[0] for i in tumorCluster])
  m2 = np.array([i[1] for i in tumorCluster])
  xmin = m1.min()
  xmax = m1.max()
  ymin = m2.min()
  ymax = m2.max()

  x, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
  positions = np.vstack([x.ravel(), Y.ravel()])
  values = np.vstack([m1, m2])
  kernel = stats.gaussian_kde(values)
  Z = np.reshape(kernel(positions).T, x.shape)


  fig, ax = plt.subplots()
  ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
            extent=[xmin, xmax, ymin, ymax])
  ax.plot(m1, m2, 'k.', markersize=2)
  ax.set_xlim([xmin, xmax])
  ax.set_ylim([ymin, ymax])
  plt.show()
  plt.savefig(outPrefix + '.png', dpi=80)


  #from ot.sliced import sliced_wasserstein_distance
  EMD = ot.sliced_wasserstein_distance(tumorCluster, lymphoCluster)

  tumorShape = alphashape.alphashape(features, 0.00625)
  lymphoShape = alphashape.alphashape(featuresLympho, 0.00625)
  intersection = tumorShape.intersection(lymphoShape).area
  union = tumorShape.area + lymphoShape.area - intersection
  jaccardDistance = intersection/union
  tumorDensity = len(features)/tumorShape.area
  lymphoDensity = len(featuresLympho)/lymphoShape.area
  print("Jaccard Distance: {0}".format(jaccardDistance))
  print("Tumor and Lympho Density: {0} and {1}".format(tumorDensity, lymphoDensity))

  Z = Z.tolist()

  return DH, Z, np.mean(Z), EMD, jaccardDistance, tumorDensity, lymphoDensity, hausdorff_dist, wasserstein, mean_log_density, js_divergence


import glob

results = {}

# directory = '/'
# directory = 'data/model.output/'  
# for filename in glob.glob(directory + '*.pt'):

inFile, outPrefix = sys.argv[1:]
print("Process: ", inFile)
model = torch.load(inFile)
try:
    DH, Z, Z_mean, EMD, jaccardDistance, tumorDensity, lymphoDensity, hausdorff_dist, wasserstein, mean_log_density, js_divergence = stat(model, outPrefix)
    results[os.path.basename(inFile)] = {'hausdorff': DH, 'kde': Z, 'mean_kde': Z_mean, 'emd': EMD, 'jaccard': jaccardDistance, 'tumorDensity': tumorDensity, 'lymphoDensity': lymphoDensity, 'max_hausdorff': hausdorff_dist, 'wasserstein': wasserstein, 'log_density': mean_log_density, 'divergence': js_divergence}
    print("Finish: ", inFile)
except Exception as e:
    print("Error: ", e)

print(results)

import json

with open(outPrefix + ".json", "w") as outfile:
    json.dump(results, outfile, indent = 2)

# results
