package com.starkey.nlusample

import android.app.Application
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import androidx.core.content.ContextCompat
import androidx.hilt.work.HiltWorkerFactory
import androidx.work.Configuration
import com.starkey.nlusample.pack.PackRepository
import com.starkey.nlusample.pack.PackSyncWorker
import dagger.hilt.android.HiltAndroidApp
import timber.log.Timber
import java.util.Locale
import javax.inject.Inject

/**
 * Wiring, and nothing else.
 *
 * The whole flow in one place:
 *
 *   Application       -> schedules the sync worker, watches the locale
 *   PackSyncWorker    -> PackRepository.sync()
 *   PackRepository    -> PackDownloader (bytes) + PackInstaller (verify+swap)
 *   AppPackProvider   -> hands the NLU module a PackSource
 *   PackIntegrity     -> the only thing that says a pack is trustworthy
 *
 * Note what the NLU module is not given: a URL, an HTTP client, an auth token,
 * or a policy about cellular data.
 */
@HiltAndroidApp
class SampleApp : Application(), Configuration.Provider {

    @Inject lateinit var workerFactory: HiltWorkerFactory
    @Inject lateinit var repository: PackRepository

    override val workManagerConfiguration: Configuration
        get() = Configuration.Builder().setWorkerFactory(workerFactory).build()

    private var localeReceiver: BroadcastReceiver? = null

    override fun onCreate() {
        super.onCreate()
        if (BuildConfig.DEBUG) Timber.plant(Timber.DebugTree())

        PackSyncWorker.schedule(this)
        registerLocaleReceiver()
    }

    /**
     * A language change invalidates the loaded pack whether or not anything is
     * downloadable.
     *
     * Registered unconditionally for that reason. The previous Android code put
     * this behind "is the downloader enabled", which meant that in the shipping
     * configuration -- no download URL, seed pack only, the default -- changing
     * the device language did nothing at all, and the engine kept serving the
     * previous language.
     */
    private fun registerLocaleReceiver() {
        if (localeReceiver != null) return
        var lastLanguage = Locale.getDefault().language

        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                val now = Locale.getDefault().language
                // ACTION_LOCALE_CHANGED also fires for region and formatting
                // changes (en-US -> en-GB), which do not change which pack we
                // load. Bumping the revision for those tears down and rebuilds
                // the classifier session for nothing.
                if (now == lastLanguage) return
                lastLanguage = now
                Timber.d("SampleApp: language -> %s, invalidating pack", now)

                // Invalidate FIRST: the pack for the new language may already
                // be installed, in which case consumers switch immediately
                // rather than waiting on a network call.
                repository.invalidate()
            }
        }
        localeReceiver = receiver

        // ACTION_LOCALE_CHANGED is a protected system broadcast, so
        // RECEIVER_NOT_EXPORTED is correct. The flag is mandatory on API 33+.
        ContextCompat.registerReceiver(
            this,
            receiver,
            IntentFilter(Intent.ACTION_LOCALE_CHANGED),
            ContextCompat.RECEIVER_NOT_EXPORTED,
        )
    }
}
