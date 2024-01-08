// Get all tabs and content elements
const tabs = document.querySelectorAll('.tab');
const contents = document.querySelectorAll('.tab-content');

// Function to remove active classes
function removeActiveClasses() {
  tabs.forEach(tab => tab.classList.remove('active'));
  contents.forEach(content => content.classList.remove('active'));
}

// Add click event to tabs
tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    removeActiveClasses();
    // Add active class to the clicked tab and the corresponding content
    tab.classList.add('active');
    const content = document.querySelector(`#${tab.id}-content`);
    content.classList.add('active');
  });
});

// Set the default active tab when the page loads
document.addEventListener('DOMContentLoaded', function() {
  removeActiveClasses(); // Ensure no tabs are active
  tabs[0].classList.add('active'); // Make the first tab active
  contents[0].classList.add('active'); // Display the first tab content
});

function startCameraStreams() {
    for (let cameraIndex = 0; cameraIndex < 2; cameraIndex++) {
        const imgElement = document.getElementById(`camera-feed-${cameraIndex}`);
        fetch(`/start_camera/${cameraIndex}`)
            .then(response => response.text())
            .then(data => {
                imgElement.src = `/video_feed/${cameraIndex}`;
                imgElement.style.display = 'block';
            })
            .catch(error => console.error('Error starting camera:', error));
    }
}

function startRecordingOn1() {
    fetch('/start_recording/0')
        .then(response => response.text())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
}

function startRecordingOn2() {
    fetch('/start_recording/1')
        .then(response => response.text())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
}

function stopRecordingOn1() {
    fetch('/stop_recording/0')
        .then(response => response.text())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
}

function stopRecordingOn2() {
    fetch('/stop_recording/1')
        .then(response => response.text())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
}

function startRecording() {
    startRecordingOn1();
    startRecordingOn2();
}

function stopRecording() {
    stopRecordingOn1();
    stopRecordingOn2();
}

window.onload = function() {
    startCameraStreams();
};

function fetchAndCreateChart(chartId, dataLabel, dataKey, unit) {
    $.ajax({
        url: '/get-sensors-input-data', // The route we created in Flask
        type: 'GET',
        dataType: 'json',
        success: function(response) {
            let labels = response.map(entry => new Date(entry.Date));
            let data = response.map(entry => entry[dataKey]);

            let ctx = document.getElementById(chartId).getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: dataLabel,
                        data: data,
                        fill: false,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: dataLabel, // Replace with your title
                            font: {
                                size: 28, // Font size for the title
                                family: 'Arial', // Optional: Font family for the title
                                weight: 'bold', // Optional: Font weight for the title
                                color: '#FF0000' // Font color for the title
                            }
                        }
                    },
                    scales: {
                        x: {
                            display: false // Hides the x-axis
                        },
                        y: {
                            ticks: {
                                beginAtZero: true,
                                font: {
                                    size: 32, // Font size for y-axis labels
                                    family: 'Arial', // Optional: Font family for y-axis labels
                                    color: '#00FF00' // Font color for y-axis labels
                                },
                                stepSize: 5
                            },
                            title: {
                                display: true,
                                text: unit,
                                font: {
                                    size: 28, // Font size for y-axis title
                                    color: '#0000FF' // Font color for y-axis title
                                }
                            }
                        }
                    }
                }                                
            });
        }
    });
}

document.getElementById('searchInput').addEventListener('keyup', function() {
    let searchQuery = this.value.toLowerCase();
    let table = document.getElementById('data-table');
    let tr = table.getElementsByTagName('tr');

    for (let i = 1; i < tr.length; i++) { // Start from 1 to skip header row
        let tds = tr[i].getElementsByTagName('td');
        let found = false;
        for (let j = 0; j < tds.length; j++) {
            if (tds[j].textContent.toLowerCase().indexOf(searchQuery) > -1) {
                found = true;
                break;
            }
        }
        tr[i].style.display = found ? "" : "none";
    }
});

function sortTable(columnIndex) {
    let table, rows, switching, i, x, y, shouldSwitch;
    table = document.getElementById("data-table");
    switching = true;
    while (switching) {
        switching = false;
        rows = table.rows;
        for (i = 1; i < (rows.length - 1); i++) { // Start from 1 to skip header row
            shouldSwitch = false;
            x = rows[i].getElementsByTagName("TD")[columnIndex];
            y = rows[i + 1].getElementsByTagName("TD")[columnIndex];
            if (x.innerHTML.toLowerCase() > y.innerHTML.toLowerCase()) {
                shouldSwitch = true;
                break;
            }
        }
        if (shouldSwitch) {
            rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
            switching = true;
        }
    }
}

function updateTable(tableId, apiUrl) {
    $.ajax({
        url: apiUrl,
        type: 'GET',
        dataType: 'json',
        success: function(response) {
            let table = document.getElementById(tableId);
            response.forEach(entry => {
                let row = table.insertRow();
                // Order of these lines should match the order of your <th> elements
                row.insertCell().textContent = entry.Date;
                row.insertCell().textContent = entry.Depth;
                row.insertCell().textContent = entry.OTemp;
                row.insertCell().textContent = entry.TTube;
                row.insertCell().textContent = entry.Humidity;
                row.insertCell().textContent = entry.Pressure;
                row.insertCell().textContent = entry.Voltage;
                row.insertCell().textContent = entry.Current;
                row.insertCell().textContent = entry.B1Voltage;
                row.insertCell().textContent = entry.B2Voltage;
                row.insertCell().textContent = entry.B3Voltage;
                row.insertCell().textContent = entry.B1Current;
                row.insertCell().textContent = entry.B2Current;
                row.insertCell().textContent = entry.B3Current;
            });
        }
    });
}

// Call updateTable for the data table
updateTable('data-table', '/get-sensors-input-data');

// Call the function for each chart when the document is ready
$(document).ready(function() {
    fetchAndCreateChart('chartDepth', 'Depth', 'Depth', 'm');
    fetchAndCreateChart('chartBattery', 'Battery Voltage', 'Voltage', 'V');
    fetchAndCreateChart('chartCurrent', 'Battery Current', 'Current', 'A');
    
    fetchAndCreateChart('chartDepth1', 'Depth', 'Depth', 'm');
    fetchAndCreateChart('chartOTemp', 'Orin Temperature', 'OTemp', '°C');
    fetchAndCreateChart('chartTTemp', 'Tube Temperature', 'TTube', '°C');
    fetchAndCreateChart('chartHumidity', 'Tube Humidity', 'Humidity', '%');
    fetchAndCreateChart('chartPressure', 'External Pressure', 'Pressure', 'psi');
    fetchAndCreateChart('chartVoltage', 'Average Voltage', 'Voltage', 'V');
    fetchAndCreateChart('chartCurrent1', 'Average Current', 'Current', 'A');
    fetchAndCreateChart('chartB1Voltage', 'B1 Voltage', 'B1Voltage', 'V');
    fetchAndCreateChart('chartB2Voltage', 'B2 Voltage', 'B2Voltage', 'V');
    fetchAndCreateChart('chartB3Voltage', 'B3 Voltage', 'B3Voltage', 'V');
    fetchAndCreateChart('chartB1Current', 'B1 Current', 'B1Current', 'A');
    fetchAndCreateChart('chartB2Current', 'B2 Current', 'B2Current', 'A');
    fetchAndCreateChart('chartB3Current', 'B3 Current', 'B3Current', 'A');
});

document.addEventListener('DOMContentLoaded', function() {
    loadFileTree();
    setInterval(function() {
        // Refresh the open file every second (or other interval) to get new log content
        let openFilePath = sessionStorage.getItem('openFilePath');
        if (openFilePath) {
            loadFileContent(openFilePath);
        }
    }, 1000);
});

function loadFileTree() {
    // Make an AJAX call to the server to get the file list
    fetch('/get-file-tree').then(response => response.json()).then(data => {
        const fileTree = document.getElementById('fileTree');
        fileTree.innerHTML = ''; // Clear previous content

        data.files.forEach(file => {
            const fileButton = document.createElement('button');
            fileButton.textContent = file;
            fileButton.classList.add('file-button'); // Add CSS class for styling
            fileButton.addEventListener('click', function() {
                loadFileContent(file);
            });
            fileTree.appendChild(fileButton);
        });
    });
}

function loadFileContent(filePath) {
    // Make an AJAX call to the server to get file content
    fetch(`/get-file-content?file=${filePath}`).then(response => response.text()).then(data => {
        const fileViewer = document.getElementById('fileViewer');
        // Replace newline characters with <br> tags
        fileViewer.innerHTML = data.replace(/\n/g, '<br>');
        sessionStorage.setItem('openFilePath', filePath); // Store the open file path
    });
}

