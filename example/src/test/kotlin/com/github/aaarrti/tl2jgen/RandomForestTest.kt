package com.github.aaarrti.tl2jgen

import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertEquals

import com.github.aaarrti.tl2jgen.randomforest.TreeEnsemble as RandomForest


class RandomForestTest {


    companion object {

        private lateinit var data: List<Pair<List<Double>, Double>>

        @BeforeAll
        @JvmStatic
        fun beforeAll() {
            data = ResourceReader.readResourceFile("random_forest.json")
        }

    }


    @TestFactory
    fun testFactory() = data.withIndex().map { (id, entry) ->
        DynamicTest.dynamicTest(id.toString()) {
            val (x, y) = entry
            val result = RandomForest.predict(x.toDoubleArray())
            assertEquals(y, result, 0.01)
        }

    }

}