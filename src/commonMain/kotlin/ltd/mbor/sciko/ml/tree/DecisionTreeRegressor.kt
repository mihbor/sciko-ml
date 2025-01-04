package ltd.mbor.sciko.ml.tree

import ltd.mbor.sciko.ml.dataframe.DataFrame
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.math.pow

class DecisionTreeRegressor {

  private var root: Node? = null

  fun fit(X: MultiArray<Double, D2>, y: MultiArray<Double, D1>): DecisionTreeRegressor {
    root = buildTree(X, y)
    return this
  }

  fun fit(X: DataFrame, y: DataFrame.Column<Double>): DecisionTreeRegressor {
    val XArray = mk.ndarray(X.columns.map { (it as DataFrame.Column<Double>).values }).transpose()
    val yArray = mk.ndarray(y.values)
    return fit(XArray, yArray)
  }

  fun predict(X: MultiArray<Double, D2>): MultiArray<Double, D1> {
    return mk.ndarray(X.toD1List().map { predict(it) })
  }

  fun predict(X: DataFrame): MultiArray<Double, D1> {
    val XArray = mk.ndarray(X.columns.map { (it as DataFrame.Column<Double>).values }).transpose()
    return predict(XArray)
  }

  fun predict(x: MultiArray<Double, D1>): Double {
    return predict(x.toList())
  }

  fun predict(x: List<Double>): Double {
    var node = root
    while (node?.left != null && node.right != null) {
      node = if (x[node.featureIndex] <= node.threshold) node.left else node.right
    }
    return node?.value ?: throw IllegalStateException("Tree is not trained.")
  }

  private fun buildTree(X: MultiArray<Double, D2>, y: MultiArray<Double, D1>, depth: Int = 0): Node {
    if (depth == MAX_DEPTH || y.toList().distinct().size == 1) {
      return Node(value = y.toList().average())
    }

    val (bestFeature, bestThreshold) = findBestSplit(X, y)
    val (leftX, leftY, rightX, rightY) = split(X, y, bestFeature, bestThreshold)

    return Node(
      featureIndex = bestFeature,
      threshold = bestThreshold,
      left = buildTree(leftX, leftY, depth + 1),
      right = buildTree(rightX, rightY, depth + 1)
    )
  }

  private fun findBestSplit(X: MultiArray<Double, D2>, y: MultiArray<Double, D1>): Pair<Int, Double> {
    var bestFeature = -1
    var bestThreshold = Double.MAX_VALUE
    var bestMSE = Double.MAX_VALUE

    for (feature in 0 until X.shape[1]) {
      val thresholds = X[feature].toList().distinct()
      for (threshold in thresholds) {
        val (leftY, rightY) = splitY(y, X[feature], threshold)
        val mse = calculateMSE(leftY, rightY)
        if (mse < bestMSE) {
          bestMSE = mse
          bestFeature = feature
          bestThreshold = threshold
        }
      }
    }

    return Pair(bestFeature, bestThreshold)
  }

  private fun split(X: MultiArray<Double, D2>, y: MultiArray<Double, D1>, feature: Int, threshold: Double): Quad<MultiArray<Double, D2>, MultiArray<Double, D1>, MultiArray<Double, D2>, MultiArray<Double, D1>> {
    val leftX = mutableListOf<List<Double>>()
    val leftY = mutableListOf<Double>()
    val rightX = mutableListOf<List<Double>>()
    val rightY = mutableListOf<Double>()

    for (i in 0 until X.shape[0]) {
      if (X[i, feature] <= threshold) {
        leftX.add(X[i].toList())
        leftY.add(y[i])
      } else {
        rightX.add(X[i].toList())
        rightY.add(y[i])
      }
    }

    return Quad(
      mk.ndarray(leftX),
      mk.ndarray(leftY),
      mk.ndarray(rightX),
      mk.ndarray(rightY)
    )
  }

  private fun splitY(y: MultiArray<Double, D1>, featureColumn: MultiArray<Double, D1>, threshold: Double): Pair<List<Double>, List<Double>> {
    val leftY = mutableListOf<Double>()
    val rightY = mutableListOf<Double>()

    for (i in 0 until featureColumn.size) {
      if (featureColumn[i] <= threshold) {
        leftY.add(y[i])
      } else {
        rightY.add(y[i])
      }
    }

    return Pair(leftY, rightY)
  }

  private fun calculateMSE(leftY: List<Double>, rightY: List<Double>): Double {
    val leftMSE = if (leftY.isEmpty()) 0.0 else leftY.map { (it - leftY.average()).pow(2) }.average()
    val rightMSE = if (rightY.isEmpty()) 0.0 else rightY.map { (it - rightY.average()).pow(2) }.average()
    return leftMSE + rightMSE
  }

  private data class Node(
    val featureIndex: Int = -1,
    val threshold: Double = Double.MAX_VALUE,
    val left: Node? = null,
    val right: Node? = null,
    val value: Double = Double.NaN
  )

  companion object {
    private const val MAX_DEPTH = 10
  }
}

private fun <T> MultiArray<T, D2>.toD1List(): List<MultiArray<T, D1>> {
  return (0..<shape[0]).map { get(it) }
}
