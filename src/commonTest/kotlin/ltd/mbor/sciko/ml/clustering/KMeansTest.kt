package ltd.mbor.sciko.ml.clustering

import ltd.mbor.sciko.ml.datasets.makeBlobs
import org.jetbrains.kotlinx.multik.ndarray.data.get
import kotlin.test.Test

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