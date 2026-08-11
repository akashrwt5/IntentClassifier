package com.starkey.nlusample.di

import com.starkey.device.features.voiceaikit.nlupack.PackProvider
import com.starkey.device.features.voiceaikit.nlupack.PackTrustPolicy
import com.starkey.nlusample.BuildConfig
import com.starkey.nlusample.pack.AppPackProvider
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import okhttp3.OkHttpClient
import java.util.concurrent.TimeUnit
import javax.inject.Singleton

/**
 * The seam. Everything above this line is the app's; everything the NLU module
 * sees is a [PackProvider] and a [PackTrustPolicy].
 */
@Module
@InstallIn(SingletonComponent::class)
object PackModule {

    @Provides
    @Singleton
    fun packProvider(impl: AppPackProvider): PackProvider = impl

    /**
     * Signing keys are the APP's, not the module's, so rotating a key is an app
     * release and not an NLU release.
     *
     * `refusesDevelopmentPacks` is tied to the build type rather than a flag
     * someone can flip. A release build cannot be talked into accepting a
     * dev-signed pack, because there is no runtime path that sets it false.
     */
    @Provides
    @Singleton
    fun packTrustPolicy(): PackTrustPolicy = PackTrustPolicy(
        publicKeys = mapOf(
            // Raw 32-byte ed25519 public key, by key_id from bundle.json.
            // Ship the production key here; the dev key only in debug builds.
            "prod-key-2026" to PRODUCTION_PUBLIC_KEY,
        ) + if (BuildConfig.DEBUG) mapOf("dev-key-golden" to DEV_PUBLIC_KEY) else emptyMap(),

        refusesDevelopmentPacks = !BuildConfig.DEBUG,
        skipsSignatureVerification = false,
    )

    @Provides
    @Singleton
    fun okHttpClient(): OkHttpClient = OkHttpClient.Builder()
        .callTimeout(2, TimeUnit.MINUTES)
        // Certificate pinning belongs here, in the app, configured once for
        // every call the app makes. A module with its own client would need its
        // own pins and would drift from them.
        .build()

    /** Replace with the real key bytes; hex here only to keep the sample readable. */
    private val PRODUCTION_PUBLIC_KEY: ByteArray = ByteArray(32)
    private val DEV_PUBLIC_KEY: ByteArray = ByteArray(32)
}
