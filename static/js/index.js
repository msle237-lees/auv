function showDiv(divId) {
    // Hide all divs
    document.getElementById('camera-viewer').style.display = 'none';
    document.getElementById('data-viewer').style.display = 'none';
    document.getElementById('log-viewer').style.display = 'none';

    // Show selected div
    document.getElementById(divId).style.display = 'flex';
}

function toggleCamera(cameraIndex) {
    const imgElement = document.getElementById(`camera-feed-${cameraIndex}`);
    const toggleButton = document.getElementById(`toggle-camera-${cameraIndex}`);

    if (imgElement.style.display === 'none' || imgElement.style.display === '') {
        // Start the camera stream
        fetch(`/start_camera/${cameraIndex}`)
            .then(response => response.text())
            .then(data => {
                imgElement.src = `/video_feed/${cameraIndex}`;
                imgElement.style.display = 'block';
                toggleButton.textContent = `Stop Camera ${cameraIndex + 1}`;
            });
    } else {
        // Stop the camera stream
        fetch(`/stop_camera/${cameraIndex}`)
            .then(response => response.text())
            .then(data => {
                imgElement.src = '';
                imgElement.style.display = 'none';
                toggleButton.textContent = `Start Camera ${cameraIndex + 1}`;
            });
    }
}

function changeCameraSource(cameraIndex) {
    const sourceInput = document.getElementById(`camera${cameraIndex + 1}-source`);
    const imgElement = document.getElementById(`camera-feed-${cameraIndex}`);
    if (sourceInput.value) {
        imgElement.src = sourceInput.value;
        imgElement.style.display = 'block';
    } else {
        alert('Please enter a valid camera source URL.');
    }
}

function openTab(evt, tabName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tab-button");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}

function showLog(logName) {
    fetch(`/get_log/${logName}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.text();
        })
        .then(data => {
            document.querySelector('.log-content').textContent = data;
        })
        .catch(error => {
            console.error('There has been a problem with your fetch operation:', error);
            document.querySelector('.log-content').textContent = 'Error loading log.';
        });
}

function startHardware() {
    fetch('/start_hardware')
        .then(response => response.text())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
}

function stopHardware() {
    fetch('/stop_hardware')
        .then(response => response.text())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
}
