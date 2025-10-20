package ltd.mbor.sciko.ml.datasets

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import kotlin.random.Random


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

fun makeMoons(
  nSamples: Int = 100,
  noise: Float? = 0.1f,
  shuffle: Boolean = true,
  seed: Long? = null
): Pair<D2Array<Float>, D1Array<Int>> {
  require(nSamples > 1) { "nSamples must be > 1" }
  val random = seed?.let { Random(it) } ?: Random.Default

  val n1 = nSamples / 2
  val n2 = nSamples - n1

  data class Sample(val x: Float, val y: Float, val label: Int)
  val samples = ArrayList<Sample>(nSamples)

  // First moon: (cos t, sin t), t in [0, pi)
  repeat(n1) {
    val t = random.nextDouble(0.0, Math.PI)
    var x = kotlin.math.cos(t).toFloat()
    var y = kotlin.math.sin(t).toFloat()
    if (noise != null && noise > 0f) {
      x += (random.nextGaussian() * noise).toFloat()
      y += (random.nextGaussian() * noise).toFloat()
    }
    samples.add(Sample(x, y, 0))
  }

  // Second moon: (1 - cos t, -sin t - 0.5), t in [0, pi)
  repeat(n2) {
    val t = random.nextDouble(0.0, Math.PI)
    var x = (1.0 - kotlin.math.cos(t)).toFloat()
    var y = (-kotlin.math.sin(t) - 0.5).toFloat()
    if (noise != null && noise > 0f) {
      x += (random.nextGaussian() * noise).toFloat()
      y += (random.nextGaussian() * noise).toFloat()
    }
    samples.add(Sample(x, y, 1))
  }

  if (shuffle) {
    // Shuffle in-place using provided RNG
    for (i in samples.lastIndex downTo 1) {
      val j = random.nextInt(i + 1)
      if (i != j) {
        val tmp = samples[i]
        samples[i] = samples[j]
        samples[j] = tmp
      }
    }
  }

  val features = FloatArray(nSamples * 2)
  val labels = IntArray(nSamples)
  for (i in 0 until nSamples) {
    val s = samples[i]
    features[2 * i] = s.x
    features[2 * i + 1] = s.y
    labels[i] = s.label
  }

  val featuresArray = mk.ndarray(features).reshape(nSamples, 2)
  val labelsArray = mk.ndarray(labels)
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