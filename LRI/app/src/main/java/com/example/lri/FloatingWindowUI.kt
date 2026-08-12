// 檔案路徑: app/src/main/java/com/example/lri/FloatingWindowUI.kt

package com.example.lri

import android.util.Log
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.RadioButtonChecked
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.github.only52607.compose.window.dragFloatingWindow

// =========================================================================
//  <<< 這裡是您所有的 UI 程式碼 >>>
// =========================================================================

@Composable
fun CollapsedView(onExpand: () -> Unit) {
    Image(
        modifier = Modifier
            .dragFloatingWindow()
            .clickable(onClick = onExpand),
        painter = painterResource(R.drawable.lri_icon),
        contentDescription = "Expand Floating Window"
    )
}

@Composable
fun ExpandedView(
    onCollapse: () -> Unit,
    onStartRecord: () -> Unit,
    onStopRecord: () -> Unit,
    onWordSplit: () -> Unit
) {
    val themeColor = Color(0xFF86AF9B)
    var isRecording by remember { mutableStateOf(false) }

    val pillTopShape = RoundedCornerShape(
        topStart = 40.dp, topEnd = 40.dp,
        bottomStart = 0.dp, bottomEnd = 0.dp
    )
    val borderColor = if (isRecording) Color(0xFFEE6969) else Color.White
    val iconSize = 30.dp

    Column(
        modifier = Modifier
            .shadow(elevation = 4.dp, shape = pillTopShape)
            .dragFloatingWindow()
            .background(themeColor, pillTopShape)
            .padding(3.dp)
            .border(3.dp, borderColor, pillTopShape)
            .clip(pillTopShape)
            .padding(vertical = 10.dp, horizontal = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        IconButton(
            onClick = onCollapse,
            modifier = Modifier.size(iconSize + 6.dp)
        ) {
            Icon(
                Icons.Default.Close,
                contentDescription = "Collapse",
                tint = Color.White,
                modifier = Modifier.size(iconSize)
            )
        }

        if (!isRecording) {
            CustomActionButton(
                onClick = {
                    Log.d("FloatingWindow", "開始錄製...")
                    isRecording = true
                    onStartRecord()
                },
                icon = Icons.Default.RadioButtonChecked,
                text = "Record",
                textColor = themeColor,
                iconSize = iconSize
            )
        } else {
            CustomActionButton(
                onClick = {
                    Log.d("FloatingWindow", "停止錄製！")
                    isRecording = false
                    onStopRecord()
                },
                icon = Icons.Filled.Stop,
                text = "Stop",
                textColor = themeColor,
                iconSize = iconSize
            )
        }

        CustomActionButton(
            onClick = onWordSplit,
            icon = Icons.Default.KeyboardArrowDown,
            text = "Next\nCharacter",
            textColor = themeColor,
            iconSize = iconSize
        )
    }
}

@Composable
fun CustomActionButton(
    onClick: () -> Unit,
    icon: ImageVector,
    text: String,
    textColor: Color,
    iconSize: Dp = 24.dp
) {
    Column(
        modifier = Modifier
            .width(55.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(Color.White)
            .clickable(onClick = onClick)
            .padding(vertical = 50.dp, horizontal = 5.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            imageVector = icon,
            contentDescription = text,
            tint = textColor,
            modifier = Modifier.size(iconSize)
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(text = text, color = textColor, fontSize = 10.sp, textAlign = TextAlign.Center)
    }
}

@Composable
fun CandidateButtons(
    candidates: List<String>,
    onCandidateSelected: (String) -> Unit
) {
    Column(
        modifier = Modifier.padding(start = 8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        candidates.forEach { text ->
            Text(
                text = text,
                fontSize = 16.sp,
                color = Color.Black,
                modifier = Modifier
                    .background(Color(0xFFE0E0E0), RoundedCornerShape(8.dp))
                    .clip(RoundedCornerShape(8.dp))
                    .clickable { onCandidateSelected(text) }
                    .padding(horizontal = 22.dp, vertical = 14.dp)
            )
        }
    }
}