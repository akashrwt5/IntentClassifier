package com.starkey.nlusample.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.starkey.nlusample.pack.PackRepository
import com.starkey.nlusample.pack.PackState
import dagger.hilt.android.AndroidEntryPoint
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.Locale
import javax.inject.Inject

/**
 * The sample's whole UI: show which pack is live and let a tester force a sync.
 *
 * It exists to make the states VISIBLE. The failure this architecture is built
 * against -- a stale pack quietly serving every turn -- has no symptom by
 * definition, so the states have to be somewhere a human can see them.
 */
@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(Modifier.fillMaxSize()) { PackScreen() }
            }
        }
    }
}

@HiltViewModel
class PackViewModel @Inject constructor(
    private val repository: PackRepository,
) : ViewModel() {

    val state = repository.state
    val revision = repository.packRevision

    fun syncNow() {
        viewModelScope.launch(Dispatchers.IO) {
            repository.sync(Locale.getDefault().language)
        }
    }
}

@Composable
fun PackScreen(vm: PackViewModel = hiltViewModel()) {
    val state by vm.state.collectAsState()
    val revision by vm.revision.collectAsState()

    Column(Modifier.padding(24.dp)) {
        Text("NLU pack", style = MaterialTheme.typography.headlineSmall)
        Spacer(Modifier.height(16.dp))

        Text(
            when (val s = state) {
                is PackState.Idle -> "idle"
                is PackState.SeedOnly -> "seed pack only (no download URL configured)"
                is PackState.Downloading -> "downloading ${s.language}…"
                is PackState.Verifying -> "verifying ${s.language}…"
                is PackState.UpToDate -> "${s.language} already up to date"
                is PackState.Installed -> "${s.language} installed and verified"
                // The interesting one. A refusal is a normal outcome, and the
                // reason is the first thing anyone debugging will want.
                is PackState.Refused -> "REFUSED ${s.language}\n${s.reason}"
                is PackState.Failed -> "download failed ${s.language}\n${s.reason}"
            }
        )

        Spacer(Modifier.height(8.dp))
        Text("packRevision = $revision", style = MaterialTheme.typography.bodySmall)
        Spacer(Modifier.height(24.dp))
        Button(onClick = vm::syncNow) { Text("Sync now") }
    }
}
