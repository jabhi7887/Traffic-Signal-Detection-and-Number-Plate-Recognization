document.addEventListener('DOMContentLoaded', function() {
    const detectedPlatesList = document.getElementById('detectedPlatesList');
    const ticketForm = document.getElementById('ticketForm');

    function fetchDetectedPlates() {
        fetch('/process_video')
            .then(response => response.json())
            .then(data => {
                detectedPlatesList.innerHTML = '';
                data.forEach(item => {
                    const li = document.createElement('li');
                    li.innerHTML = `<strong>${item.text}</strong><br><img src="data:image/jpeg;base64,${item.image}" width="200" />`;
                    detectedPlatesList.appendChild(li);
                });
            })
            .catch(error => console.error('Error fetching detected plates:', error));
    }

    setInterval(fetchDetectedPlates, 5000);

    ticketForm.addEventListener('submit', function(event) {
        event.preventDefault();

        const formData = {
            plate_number: ticketForm.plateNumber.value,
            camera_id: ticketForm.cameraId.value,
            location: ticketForm.location.value,
            detected_image: ticketForm.detectedImage.value
        };

        fetch('/issue_ticket', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        })
        .then(response => response.json())
        .then(data => {
            alert('Ticket issued successfully!');
            ticketForm.reset();
            fetchDetectedPlates();
        })
        .catch(error => console.error('Error issuing ticket:', error));
    });
});
