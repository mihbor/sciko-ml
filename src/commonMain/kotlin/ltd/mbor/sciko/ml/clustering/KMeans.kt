package ltd.mbor.sciko.ml.clustering

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.sqrt
import kotlin.random.Random

class KMeans(
  private val k: Int,
  private val maxIterations: Int = 300,
  private val tolerance: Float = 1e-4f,
  private val seed: Long? = null
) {
  private val random = seed?.let { Random(it) } ?: Random.Default

  fun fit(data: MultiArray<Float, D2>): KMeansResult {
    val nSamples = data.shape[0]
    val nFeatures = data.shape[1]

    // Initialize centroids randomly from data points
    val indices = (0 until nSamples).shuffled(random).take(k)
    var centroids: MultiArray<Float, D2> = mk.stack(indices.map { data[it] })
    var labels: MultiArray<Int, D1> = mk.zeros<Int>(nSamples)

    repeat(maxIterations) { iteration ->
      // Assign each point to nearest centroid
      val newLabels = assignToNearestCentroids(data, centroids)

      // Check for convergence
      if (iteration > 0 && labelsEqual(labels, newLabels)) {
        labels = newLabels
        return@repeat
      }

      labels = newLabels

      // Update centroids
      val oldCentroids = centroids.deepCopy()
      centroids = updateCentroids(data, labels, nFeatures)

      // Check centroid movement
      val movement = computeMaxMovement(oldCentroids, centroids)

      if (movement < tolerance) {
        return@repeat
      }
    }

    val inertia = computeInertia(data, centroids, labels)
    return KMeansResult(centroids, labels, inertia)
  }

  private fun assignToNearestCentroids(data: MultiArray<Float, D2>, centroids: MultiArray<Float, D2>): MultiArray<Int, D1> {
    val nSamples = data.shape[0]
    val labels = IntArray(nSamples)

    for (i in 0 until nSamples) {
      val point = data[i]
      var minDist = Float.MAX_VALUE
      var minIdx = 0

      for (j in 0 until k) {
        val dist = euclideanDistance(point, centroids[j])
        if (dist < minDist) {
          minDist = dist
          minIdx = j
        }
      }
      labels[i] = minIdx
    }

    return mk.ndarray(labels)
  }

  private fun updateCentroids(data: MultiArray<Float, D2>, labels: MultiArray<Int, D1>, nFeatures: Int): MultiArray<Float, D2> {
    val newCentroids = mk.zeros<Float>(k, nFeatures)
    val counts = IntArray(k)

    // Sum points for each cluster
    for (i in 0 until data.shape[0]) {
      val cluster = labels[i]
      counts[cluster]++
      for (j in 0 until nFeatures) {
        newCentroids[cluster, j] = newCentroids[cluster, j] + data[i, j]
      }
    }

    // Average
    for (i in 0 until k) {
      if (counts[i] > 0) {
        for (j in 0 until nFeatures) {
          newCentroids[i, j] = newCentroids[i, j] / counts[i]
        }
      }
    }

    return newCentroids
  }

  private fun euclideanDistance(a: MultiArray<Float, D1>, b: MultiArray<Float, D1>): Float {
    var sum = 0f
    for (i in 0 until a.size) {
      val diff = a[i] - b[i]
      sum += diff * diff
    }
    return sqrt(sum)
  }

  private fun computeMaxMovement(oldCentroids: MultiArray<Float, D2>, newCentroids: MultiArray<Float, D2>): Float {
    var maxMovement = 0f
    for (i in 0 until k) {
      val movement = euclideanDistance(oldCentroids[i], newCentroids[i])
      if (movement > maxMovement) {
        maxMovement = movement
      }
    }
    return maxMovement
  }

  private fun computeInertia(data: MultiArray<Float, D2>, centroids: MultiArray<Float, D2>, labels: MultiArray<Int, D1>): Float {
    var inertia = 0f
    for (i in 0 until data.shape[0]) {
      val dist = euclideanDistance(data[i], centroids[labels[i]])
      inertia += dist * dist
    }
    return inertia
  }

  private fun labelsEqual(a: MultiArray<Int, D1>, b: MultiArray<Int, D1>): Boolean {
    if (a.size != b.size) return false
    for (i in 0 until a.size) {
      if (a[i] != b[i]) return false
    }
    return true
  }
}

data class KMeansResult(
  val centroids: MultiArray<Float, D2>,
  val labels: MultiArray<Int, D1>,
  val inertia: Float
)