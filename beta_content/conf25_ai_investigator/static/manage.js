// Global error handler
window.onerror = function(message, source, lineno, colno, error) {
    console.error('Global error:', message, 'at', source, 'line', lineno, 'column', colno, 'Error object:', error);
    showNotification('An unexpected error occurred. Please check the console for more details.', 'error');
    return false;
};

// Function to handle upload type selection
function selectUploadType(type, clickedButton) {
    try {
        // Set the hidden input value for the upload type
        const uploadTypeInput = document.getElementById('upload-type');
        if (uploadTypeInput) {
            uploadTypeInput.value = type;
        }

        // Update button styles
        const buttons = document.querySelectorAll('.upload-type-buttons .btn');
        buttons.forEach(btn => btn.classList.remove('btn-active'));
        clickedButton.classList.add('btn-active');

        // Show/hide data description fields
        const dataDescriptionFields = document.getElementById('data-description-fields');
        if (dataDescriptionFields) {
            dataDescriptionFields.style.display = type === 'data_description' ? 'block' : 'none';
        }
    } catch (error) {
        console.error('Error in selectUploadType:', error);
        showNotification('An error occurred while selecting upload type.', 'error');
    }
}

// Function to show modal with formatted content
function showModal(modalId) {
    try {
        const modal = document.getElementById(modalId);
        if (!modal) return;

        const contentElement = modal.querySelector('.modal-content');
        if (!contentElement) return;

        // Get the entry type from the data attribute
        const entryType = contentElement.getAttribute('data-entry-type');

        // Get the raw content
        const rawContentElement = contentElement.querySelector('pre');
        if (!rawContentElement) return;
        const rawContent = rawContentElement.textContent.trim();
        
        // Find or create the modal-body
        let modalBody = contentElement.querySelector('.modal-body');
        if (!modalBody) {
            modalBody = document.createElement('div');
            modalBody.className = 'modal-body';
            contentElement.appendChild(modalBody);
        }
        
        // Clear existing formatted content
        const existingFormattedContent = modalBody.querySelector('.formatted-content');
        if (existingFormattedContent) {
            existingFormattedContent.remove();
        }
        
        // Add new formatted content based on entry type
        if (entryType === 'data_description') {
            const description = contentElement.getAttribute('data-description');
            modalBody.appendChild(displayDataDescription(rawContent, description));
        } else if (entryType === 'tool_description') {
            // Add specific handling for tool descriptions
            const formattedContent = document.createElement('div');
            formattedContent.className = 'formatted-content tool-description-content';
            
            const toolNameDiv = document.createElement('div');
            toolNameDiv.className = 'tool-description-row';
            toolNameDiv.innerHTML = `<strong>Tool Name:</strong> <span>${contentElement.getAttribute('data-tool-name') || 'N/A'}</span>`;
            formattedContent.appendChild(toolNameDiv);
            
            const descriptionDiv = document.createElement('div');
            descriptionDiv.className = 'tool-description-row';
            descriptionDiv.innerHTML = `<strong>Description:</strong> <span>${rawContent}</span>`;
            formattedContent.appendChild(descriptionDiv);
            
            const thresholdDiv = document.createElement('div');
            thresholdDiv.className = 'tool-description-row';
            thresholdDiv.innerHTML = `<strong>Threshold:</strong> <span>${contentElement.getAttribute('data-threshold') || 'N/A'}</span>`;
            formattedContent.appendChild(thresholdDiv);
            
            modalBody.appendChild(formattedContent);
        } else {
            modalBody.appendChild(displayContent(rawContent));
        }
        
        modal.style.display = 'block';
    } catch (error) {
        console.error('Error in showModal:', error);
        showNotification('An error occurred while showing the modal.', 'error');
    }
}

// Function to hide modal
function hideModal(modalId) {
    try {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.style.display = 'none';
        }
    } catch (error) {
        console.error('Error in hideModal:', error);
        showNotification('An error occurred while hiding the modal.', 'error');
    }
}

// Function to handle entry removal
function removeEntry(id) {
    try {
        if (confirm('Are you sure you want to remove this entry?')) {
            fetch(`/remove_entry/${id}`, { 
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                },
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                showNotification(data.message, 'success');
                const row = document.querySelector(`tr[data-id="${id}"]`);
                if (row) {
                    row.remove();
                } else {
                    location.reload(); // Fallback to page reload if row not found
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showNotification('An error occurred while trying to remove the entry.', 'error');
            });
        }
    } catch (error) {
        console.error('Error in removeEntry:', error);
        showNotification('An error occurred while removing the entry.', 'error');
    }
}

// Function to handle form submission for adding content
function handleAddContent(event) {
    try {
        event.preventDefault();
        const form = document.getElementById('add-content-form');
        if (!form) return;

        const formData = new FormData(form);

        // Get the selected data store value
        const dataStoreSelect = document.getElementById('data-store');
        if (dataStoreSelect) {
            formData.set('data_store', dataStoreSelect.value);
        }

        // Get the description value for data_description type
        const uploadType = formData.get('upload_type');
        if (uploadType === 'data_description') {
            const descriptionTextarea = document.getElementById('description');
            if (descriptionTextarea) {
                formData.set('description', descriptionTextarea.value);
            }
        }

        fetch('/add_content', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            showNotification(data.message, 'success');
            form.reset(); // Clear the form after successful addition
            if (data.redirect) {
                window.location.href = data.redirect;
            } else {
                location.reload(); // Reload the page to reflect the changes
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification('An error occurred while processing your request: ' + error.message, 'error');
        });
    } catch (error) {
        console.error('Error in handleAddContent:', error);
        showNotification('An error occurred while adding content.', 'error');
    }
}

// Function to display content with improved color coding
function displayContent(content) {
    try {
        // Create a container for the formatted content
        let formattedContent = document.createElement('div');
        formattedContent.className = 'formatted-content';

        // Split the content into sections
        const sections = content.split(/(?=User query:|SPL answer:|Explanation:)/g);
        
        // Define colors for each section
        const colors = {
            'User query:': '#e6f3ff',  // Light blue
            'SPL answer:': '#fff0e6',  // Light orange
            'Explanation:': '#e6ffe6'  // Light green
        };

        // Process each section
        sections.forEach(section => {
            const trimmedSection = section.trim();
            const sectionDiv = document.createElement('div');
            sectionDiv.className = 'content-section';
            
            // Determine the section type and apply appropriate styling
            for (const [sectionType, color] of Object.entries(colors)) {
                if (trimmedSection.startsWith(sectionType)) {
                    sectionDiv.style.backgroundColor = color;
                    
                    // Create a bold header for the section type
                    const header = document.createElement('strong');
                    header.textContent = sectionType;
                    sectionDiv.appendChild(header);
                    
                    // Add the content
                    const content = document.createElement('div');
                    content.textContent = trimmedSection.substring(sectionType.length).trim();
                    sectionDiv.appendChild(content);
                    
                    break;
                }
            }
            
            formattedContent.appendChild(sectionDiv);
        });

        return formattedContent;
    } catch (error) {
        console.error('Error in displayContent:', error);
        showNotification('An error occurred while displaying content.', 'error');
        return document.createTextNode('Error displaying content');
    }
}

// Function to display data description content
function displayDataDescription(content, description) {
    try {
        const container = document.createElement('div');
        container.className = 'formatted-content data-description-content';

        // Add description if available
        if (description) {
            const descriptionRow = document.createElement('div');
            descriptionRow.className = 'data-description-row';
            
            const descKeyElement = document.createElement('strong');
            descKeyElement.textContent = 'Description:';
            
            const descValueElement = document.createElement('span');
            descValueElement.textContent = description;
            
            descriptionRow.appendChild(descKeyElement);
            descriptionRow.appendChild(descValueElement);
            container.appendChild(descriptionRow);
        }

        const lines = content.split('\n');
        lines.forEach(line => {
            const [key, value] = line.split(':').map(item => item.trim());
            if (key && value) {
                const row = document.createElement('div');
                row.className = 'data-description-row';
                
                const keyElement = document.createElement('strong');
                keyElement.textContent = key + ':';
                
                const valueElement = document.createElement('span');
                valueElement.textContent = value;
                
                row.appendChild(keyElement);
                row.appendChild(valueElement);
                container.appendChild(row);
            }
        });

        return container;
    } catch (error) {
        console.error('Error in displayDataDescription:', error);
        showNotification('An error occurred while displaying data description.', 'error');
        return document.createTextNode('Error displaying data description');
    }
}

// Function to show notification
function showNotification(message, type = 'info') {
    try {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        document.body.appendChild(notification);

        setTimeout(() => {
            notification.classList.add('show');
            setTimeout(() => {
                notification.classList.remove('show');
                setTimeout(() => {
                    notification.remove();
                }, 300);
            }, 3000);
        }, 100);
    } catch (error) {
        console.error('Error in showNotification:', error);
        alert('An error occurred while showing a notification: ' + message);
    }
}

// Event listener for page load
document.addEventListener('DOMContentLoaded', function() {
    try {
        // Attach event listeners to view buttons
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const modalId = btn.getAttribute('data-modal-id');
                showModal(modalId);
            });
        });

        // Attach event listeners to remove buttons
        document.querySelectorAll('.remove-btn').forEach(btn => {
            btn.addEventListener('click', (event) => {
                event.preventDefault(); // Prevent default button behavior
                const id = btn.getAttribute('data-id');
                removeEntry(id);
            });
        });

        // Attach event listener for adding content
        const addContentForm = document.getElementById('add-content-form');
        if (addContentForm) {
            addContentForm.addEventListener('submit', handleAddContent);
        }

        // Close modal when clicking outside of it
        window.onclick = function(event) {
            if (event.target.classList.contains('modal')) {
                event.target.style.display = 'none';
            }
        };

        // Initialize upload type buttons
        const uploadTypeButtons = document.querySelectorAll('.upload-type-buttons .btn');
        uploadTypeButtons.forEach(button => {
            button.addEventListener('click', function() {
                const type = this.getAttribute('data-type');
                selectUploadType(type, this);
            });
        });

        // Set initial upload type to "Query Example"
        const queryExampleButton = document.querySelector('.upload-type-buttons .btn[data-type="query_example"]');
        if (queryExampleButton) {
            selectUploadType('query_example', queryExampleButton);
        }

        // Add event listener for data store select
        const dataStoreSelect = document.getElementById('data-store');
        if (dataStoreSelect) {
            dataStoreSelect.addEventListener('change', function() {
                console.log('Data store selected:', this.value);
            });
        }

        console.log('DOMContentLoaded event handler completed successfully');
    } catch (error) {
        console.error('Error in DOMContentLoaded event handler:', error);
        showNotification('An error occurred while initializing the page.', 'error');
    }
});

console.log('manage.js loaded successfully');