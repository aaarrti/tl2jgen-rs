package com.github.aaarrti.tl2jgen

import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertEquals

import com.github.aaarrti.tl2jgen.xgboost.TreeEnsemble as Xgboost


class XgboostTest {


    companion object {

        private lateinit var data: List<Pair<List<Double>, Double>>

        @BeforeAll
        @JvmStatic
        fun beforeAll() {
            data = ResourceReader.readResourceFile("xgboost.json")
        }

    }


    @TestFactory
    fun testFactory() = data.withIndex().map { (id, entry) ->
        DynamicTest.dynamicTest(id.toString()) {
            val (x, y) = entry
            val result = Xgboost.predict(x.toDoubleArray())
            assertEquals(y, result, 0.01)
        }
    }

}