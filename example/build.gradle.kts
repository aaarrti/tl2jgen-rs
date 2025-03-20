import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi

plugins {
    java
    kotlin("jvm") version "2.1.10"
    id("org.jetbrains.kotlin.plugin.serialization") version "2.1.20"
    kotlin("plugin.power-assert") version "2.0.0"
}

group = "com.github.aaarrti.tl2jgen"
version = "0.0.1"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json-jvm:1.8.0")
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}

@OptIn(ExperimentalKotlinGradlePluginApi::class)
powerAssert {
    functions = listOf("kotlin.assert", "kotlin.test.assertTrue", "kotlin.test.assertEquals", "kotlin.test.assertNull")
    includedSourceSets = listOf("commonMain", "jvmMain")
}