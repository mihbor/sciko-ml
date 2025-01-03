package ltd.mbor.sciko.ml

import ltd.mbor.sciko.dataframe.DataFrame
import kotlin.test.Test
import kotlin.test.assertEquals

class SplitTest {

  @Test
  fun testTrainTestSplit() {
    val csvData = """
            name,age,active
            John,30,true
            Jane,25,false
            Bob,40,true
            Alice,35,false
        """.trimIndent()

    val dataFrame = DataFrame.readCsv(csvData)
    val (trainDataFrame, testDataFrame) = dataFrame.trainTestSplit(0.5)

    assertEquals(2, testDataFrame.rowCount)
    assertEquals(2, trainDataFrame.rowCount)

    // Check that all columns are present in both splits
    assertEquals(dataFrame.columns.size, trainDataFrame.columns.size)
    assertEquals(dataFrame.columns.size, testDataFrame.columns.size)

    // Check that the sum of rows in train and test sets equals the original row count
    assertEquals(dataFrame.rowCount, trainDataFrame.rowCount + testDataFrame.rowCount)
  }
}