package ltd.mbor.sciko.ml.dataframe

import kotlin.test.Test
import kotlin.test.assertEquals

class DataFrameTest {

  @Test
  fun testReadCsv() {
    val csvData = """
            name,age,active
            John,30,true
            Jane,25,false
        """.trimIndent()

    val dataFrame = DataFrame.readCsv(csvData)

    assertEquals(3, dataFrame.columns.size)

    val nameColumn = dataFrame.columns[0]
    assertEquals("name", nameColumn.name)
    assertEquals(DataFrame.DataType.STRING, nameColumn.type)
    assertEquals(listOf("John", "Jane"), nameColumn.values)

    val ageColumn = dataFrame.columns[1]
    assertEquals("age", ageColumn.name)
    assertEquals(DataFrame.DataType.DOUBLE, ageColumn.type)
    assertEquals(listOf(30.0, 25.0), ageColumn.values)

    val activeColumn = dataFrame.columns[2]
    assertEquals("active", activeColumn.name)
    assertEquals(DataFrame.DataType.BOOLEAN, activeColumn.type)
    assertEquals(listOf(true, false), activeColumn.values)
  }

  @Test
  fun testDropNA() {
    val csvData = """
        name,age,active
        John,30,true
        Jane,,false
        """.trimIndent()

    val dataFrame = DataFrame.readCsv(csvData)
    val cleanedDataFrame = dataFrame.dropNA(listOf("age"))

    assertEquals(3, cleanedDataFrame.columns.size)

    val nameColumn = cleanedDataFrame.columns[0]
    assertEquals("name", nameColumn.name)
    assertEquals(DataFrame.DataType.STRING, nameColumn.type)
    assertEquals(listOf("John"), nameColumn.values)

    val ageColumn = cleanedDataFrame.columns[1]
    assertEquals("age", ageColumn.name)
    assertEquals(DataFrame.DataType.DOUBLE, ageColumn.type)
    assertEquals(listOf(30.0), ageColumn.values)

    val activeColumn = cleanedDataFrame.columns[2]
    assertEquals("active", activeColumn.name)
    assertEquals(DataFrame.DataType.BOOLEAN, activeColumn.type)
    assertEquals(listOf(true), activeColumn.values)
  }
}