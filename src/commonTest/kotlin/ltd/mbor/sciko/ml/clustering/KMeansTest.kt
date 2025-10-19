package ltd.mbor.sciko.ml.clustering

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import kotlin.random.Random
import kotlin.test.Test

fun makeBlobs(
  nSamples: Int = 100,
  nFeatures: Int = 2,
  centers: Int = 3,
  clusterStd: Float = 1.0f,
  centerBox: Pair<Float, Float> = -10.0f to 10.0f,
  seed: Long? = null
): Pair<D2Array<Float>, D1Array<Int>> {
  val random = seed?.let { Random(it) } ?: Random.Default

  // Generate random centers
  val centerCoords = Array(centers) {
    FloatArray(nFeatures) {
      random.nextFloat() * (centerBox.second - centerBox.first) + centerBox.first
    }
  }

  // Calculate samples per center
  val samplesPerCenter = IntArray(centers) { nSamples / centers }
  for (i in 0 until (nSamples % centers)) {
    samplesPerCenter[i]++
  }

  val features = mutableListOf<Float>()
  val labels = mutableListOf<Int>()

  // Generate samples for each center
  for (centerIdx in 0 until centers) {
    val center = centerCoords[centerIdx]
    repeat(samplesPerCenter[centerIdx]) {
      for (featureIdx in 0 until nFeatures) {
        val gaussian = random.nextGaussian().toFloat()
        features.add(center[featureIdx] + gaussian * clusterStd)
      }
      labels.add(centerIdx)
    }
  }

  val featuresArray = mk.ndarray(features.toFloatArray()).reshape(nSamples, nFeatures)
  val labelsArray = mk.ndarray(labels.toIntArray())

  return featuresArray to labelsArray
}

fun Random.nextGaussian(): Double {
  var u1: Double
  var u2: Double
  var s: Double
  do {
    u1 = nextDouble() * 2.0 - 1.0
    u2 = nextDouble() * 2.0 - 1.0
    s = u1 * u1 + u2 * u2
  } while (s >= 1.0 || s == 0.0)
  return u1 * kotlin.math.sqrt(-2.0 * kotlin.math.ln(s) / s)
}

class KMeansTest {
  @Test
  fun testKMeans() {
    // Generate synthetic data
    val (features, trueLabels) = makeBlobs(
      nSamples = 300,
      nFeatures = 2,
      centers = 3,
      clusterStd = 0.8f,
      seed = 42L
    )

    println("Data shape: ${features.shape.contentToString()}")
    println("Labels shape: ${trueLabels.shape.contentToString()}")

    // Fit K-Means
    val kmeans = KMeans(k = 3, seed = 42L)
    val result = kmeans.fit(features)

    println("\nInertia: ${result.inertia}")
    println("First 10 cluster assignments: ${result.labels[0..9]}")
    println("\nCentroids:")
    for (i in 0 until 3) {
      println("  Cluster $i: ${result.centroids[i]}")
    }
  }
}