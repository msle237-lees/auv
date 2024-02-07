function openTab(evt, tabName) {
    // Declare all variables
    let i, tabcontent, tablinks;

    // Get all elements with class="tabcontent" and hide them
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }

    if(evt != null) {
        evt.currentTarget.className += " active";
    }

    // Get all elements with class="tablinks" and remove the class "active"
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    // Show the current tab, and add an "active" class to the button that opened the tab
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}

function openDataTab(evt, dataTabName) {
    // Declare all variables
    let i, datacontent, tablinks;

    // Get all elements with class="datacontent" and hide them
    datacontent = document.getElementsByClassName("datacontent");
    for (i = 0; i < datacontent.length; i++) {
        datacontent[i].style.display = "none";
    }

    if(evt != null) {
        evt.currentTarget.className += " active";
    }

    // Get all elements with class="tablinks" and remove the class "active"
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        if (tablinks[i].parentNode.className === "tab data-tab") {
            tablinks[i].className = tablinks[i].className.replace(" active", "");
        }
    }

    // Show the current tab, and add an "active" class to the button that opened the tab
    document.getElementById(dataTabName).style.display = "block";
    evt.currentTarget.className += " active";
}

document.getElementById('darkModeToggle').addEventListener('change', function(event) {
    if (event.target.checked) {
        document.body.classList.add('dark-mode');
        document.querySelector('.tab').classList.add('dark-mode');
        // Add dark-mode class to other elements as needed
    } else {
        document.body.classList.remove('dark-mode');
        document.querySelector('.tab').classList.remove('dark-mode');
        // Remove dark-mode class from other elements as needed
    }
});

function setDefaultMode() {
    // Set dark mode as default
    document.body.classList.add('dark-mode');
    document.querySelector('.tab').classList.add('dark-mode');
    document.getElementById('darkModeToggle').checked = true;
    // Open default external tab (Camera Feeds)
    openTab(null, 'CameraFeeds'); // Pass null for the event if it's not available
    // Open default internal tab (Input Data)
    openDataTab(null, 'InputData'); // Pass null for the event if it's not available
}
