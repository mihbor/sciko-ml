package ltd.mbor.sciko.ml.tree

import ltd.mbor.sciko.ml.dataframe.DataFrame
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

class DecisionTreeClassifier {

  private var root: Node? = null

  fun fit(X: MultiArray<Double, D2>, y: MultiArray<Int, D1>): DecisionTreeClassifier {
    root = buildTree(X, y)
    return this
  }

  fun fit(X: DataFrame, y: DataFrame.Column<Int>): DecisionTreeClassifier {
    val XArray = mk.ndarray(X.columns.map { (it as DataFrame.Column<Double>).values }).transpose()
    val yArray = mk.ndarray(y.values)
    return fit(XArray, yArray)
  }

  fun predict(X: MultiArray<Double, D2>): MultiArray<Int, D1> {
    return mk.ndarray(X.toD1List().map { predict(it) })
  }

  fun predict(X: DataFrame): MultiArray<Int, D1> {
    val XArray = mk.ndarray(X.columns.map { (it as DataFrame.Column<Double>).values }).transpose()
    return predict(XArray)
  }

  fun predict(x: MultiArray<Double, D1>): Int {
    return predict(x.toList())
  }

  fun predict(x: List<Double>): Int {
    var node = root
    while (node?.left != null && node.right != null) {
      node = if (x[node.featureIndex] <= node.threshold) node.left else node.right
    }
    return node?.value ?: throw IllegalStateException("Tree is not trained.")
  }

  private fun buildTree(X: MultiArray<Double, D2>, y: MultiArray<Int, D1>, depth: Int = 0): Node {
    if (depth == MAX_DEPTH || y.toList().distinct().size == 1) {
      return Node(value = y.toList().groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: 0)
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

  private fun findBestSplit(X: MultiArray<Double, D2>, y: MultiArray<Int, D1>): Pair<Int, Double> {
    var bestFeature = -1
    var bestThreshold = Double.MAX_VALUE
    var bestGini = Double.MAX_VALUE

    for (feature in 0 until X.shape[1]) {
      val thresholds = X[feature].toList().distinct()
      for (threshold in thresholds) {
        val (leftY, rightY) = splitY(y, X[feature], threshold)
        val gini = calculateGini(leftY, rightY)
        if (gini < bestGini) {
          bestGini = gini
          bestFeature = feature
          bestThreshold = threshold
        }
      }
    }

    return Pair(bestFeature, bestThreshold)
  }

  private fun split(X: MultiArray<Double, D2>, y: MultiArray<Int, D1>, feature: Int, threshold: Double): Quad<MultiArray<Double, D2>, MultiArray<Int, D1>, MultiArray<Double, D2>, MultiArray<Int, D1>> {
    val leftX = mutableListOf<List<Double>>()
    val leftY = mutableListOf<Int>()
    val rightX = mutableListOf<List<Double>>()
    val rightY = mutableListOf<Int>()

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

  private fun splitY(y: MultiArray<Int, D1>, featureColumn: MultiArray<Double, D1>, threshold: Double): Pair<List<Int>, List<Int>> {
    val leftY = mutableListOf<Int>()
    val rightY = mutableListOf<Int>()

    for (i in 0 until featureColumn.size) {
      if (featureColumn[i] <= threshold) {
        leftY.add(y[i])
      } else {
        rightY.add(y[i])
      }
    }

    return Pair(leftY, rightY)
  }

  private fun calculateGini(leftY: List<Int>, rightY: List<Int>): Double {
    val leftGini = if (leftY.isEmpty()) 0.0 else 1.0 - leftY.groupingBy { it }.eachCount().values.sumOf { (it.toDouble() / leftY.size).pow(2) }
    val rightGini = if (rightY.isEmpty()) 0.0 else 1.0 - rightY.groupingBy { it }.eachCount().values.sumOf { (it.toDouble() / rightY.size).pow(2) }
    return leftGini + rightGini
  }

  private data class Node(
    val featureIndex: Int = -1,
    val threshold: Double = Double.MAX_VALUE,
    val left: Node? = null,
    val right: Node? = null,
    val value: Int = Int.MIN_VALUE
  )

  companion object {
    private const val MAX_DEPTH = 10
  }
}

private fun <T> MultiArray<T, D2>.toD1List(): List<MultiArray<T, D1>> {
  return (0..<shape[0]).map { get(it) }
}

private data class Quad<A, B, C, D>(
  val first: A,
  val second: B,
  val third: C,
  val fourth: D
)
