package ltd.mbor.sciko.ml.clustering

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.get

class DBSCAN(
  private val eps: Float,
  private val minSamples: Int
) {
  init {
    require(eps > 0f) { "eps must be > 0" }
    require(minSamples >= 1) { "minSamples must be >= 1" }
  }

  private val NOISE = -1
  private val UNVISITED = Int.MIN_VALUE

  fun fit(data: D2Array<Float>): D1Array<Int> {
    val n = data.shape[0]
    if (n == 0) return mk.ndarray(IntArray(0))

    val labels = IntArray(n) { UNVISITED }
    val eps2 = eps * eps

    // Precompute neighbors for all points (naive O(n^2))
    val neighbors: Array<IntArray> = Array(n) { i ->
      val inds = ArrayList<Int>()
      for (j in 0 until n) {
        if (squaredDistance(data[i], data[j]) <= eps2) inds.add(j)
      }
      inds.toIntArray()
    }

    var clusterId = 0

    for (i in 0 until n) {
      if (labels[i] != UNVISITED) continue
      val neigh = neighbors[i]
      if (neigh.size < minSamples) {
        labels[i] = NOISE
      } else {
        expandCluster(i, neighbors, labels, clusterId)
        clusterId++
      }
    }

    // Relabel clusters to 0..k-1, keep -1 as noise
    if (clusterId > 0) {
      val map = HashMap<Int, Int>()
      var next = 0
      for (i in 0 until n) {
        val l = labels[i]
        if (l >= 0) {
          val new = map.getOrPut(l) { next++ }
          labels[i] = new
        }
      }
    }

    return mk.ndarray(labels)
  }

  private fun expandCluster(start: Int, neighbors: Array<IntArray>, labels: IntArray, clusterId: Int) {
    val queue: ArrayDeque<Int> = ArrayDeque()
    // assign start to cluster
    labels[start] = clusterId
    // seed with its neighbors
    for (nb in neighbors[start]) queue.addLast(nb)

    while (queue.isNotEmpty()) {
      val point = queue.removeFirst()
      val label = labels[point]

      if (label == NOISE) {
        labels[point] = clusterId // border point
      }
      if (labels[point] != UNVISITED && labels[point] != clusterId) {
        continue
      }
      if (labels[point] == UNVISITED) {
        labels[point] = clusterId
        val neigh = neighbors[point]
        if (neigh.size >= minSamples) {
          for (nb in neigh) {
            if (labels[nb] == UNVISITED) queue.addLast(nb)
          }
        }
      }
    }
  }

  private fun squaredDistance(a: MultiArray<Float, org.jetbrains.kotlinx.multik.ndarray.data.D1>, b: MultiArray<Float, org.jetbrains.kotlinx.multik.ndarray.data.D1>): Float {
    var sum = 0f
    for (i in 0 until a.size) {
      val d = a[i] - b[i]
      sum += d * d
    }
    return sum
  }
}
