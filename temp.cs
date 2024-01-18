using UnityEngine;
using System.Collections;
using System.IO;
using UnityEngine.Networking; // Add this namespace to use UnityWebRequest

public class RobotCamera : MonoBehaviour {
    // ... (rest of your variables)

    private string flaskEndpoint = "http://localhost:5000/upload_image"; // Flask server URL

    // ... (rest of your methods)

    private IEnumerator CaptureAndSend(Camera cam, string endpoint, int id){
        // Capture image
        RenderTexture activeRT = RenderTexture.active;
        Texture2D image = new Texture2D(cam.targetTexture.width, cam.targetTexture.height);
        cam.Render();
        RenderTexture.active = cam.targetTexture;
        image.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0,0);
        image.Apply();
        RenderTexture.active = activeRT;

        // Convert to byte array and destroy Texture2D to free memory
        byte[] data = image.EncodeToPNG();
        Destroy(image);

        // Create a WebRequest to send the image
        UnityWebRequest www = new UnityWebRequest(endpoint, "POST");
        www.uploadHandler = new UploadHandlerRaw(data);
        www.downloadHandler = new DownloadHandlerBuffer();
        www.SetRequestHeader("Content-Type", "application/octet-stream");

        // Send the request and wait for a response
        yield return www.SendWebRequest();

        if(www.isNetworkError || www.isHttpError) {
            Debug.Log(www.error);
        }
        else {
            Debug.Log("Image upload complete!");
        }
    }

    // ... (rest of your methods, e.g., CommandCapture where you'd call CaptureAndSend)

    private void CommandCapture(){
        switch (currentCommand) {
            case CamCommandsID.front_no_percept:
                StartCoroutine(CaptureAndSend(frontCamera, flaskEndpoint + "?camera=front", frontCounter));
                frontCounter++;
                break;
            case CamCommandsID.down_no_percept:
                StartCoroutine(CaptureAndSend(downCamera, flaskEndpoint + "?camera=down", downCounter));
                downCounter++;
                break;
            // ... (rest of your cases)
        }
    }

    // ... (rest of your methods)
}


private IEnumerator CaptureAndSendContinuously() {
        while (true) {
            // Assuming you want to capture and send images from both cameras continuously
            StartCoroutine(CaptureAndSend(frontCamera, flaskEndpoint + "?camera=front", frontCounter));
            frontCounter++;
            StartCoroutine(CaptureAndSend(downCamera, flaskEndpoint + "?camera=down", downCounter));
            downCounter++;

            yield return new WaitForSeconds(captureInterval);
        }
    }

    private IEnumerator CaptureAndSend(Camera cam, string endpoint, int id) {
        // ... (existing CaptureAndSend code)
    }