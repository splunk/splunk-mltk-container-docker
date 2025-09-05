// Function to handle query generation
async function generateQuery() {
    const query = document.getElementById('query').value;
    const llmProvider = document.getElementById('llm-provider').value;
    const resultDiv = document.getElementById('result');
    const splQueryPre = document.getElementById('spl-query');
    const llmQueryPre = document.getElementById('llm-query');
    const systemPromptTextarea = document.getElementById('system-prompt');
    const totalAttemptsPre = document.getElementById('total-attempts');
    const errorMessageDiv = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');
    const vizRecommendation = document.getElementById('viz-recommendation');

    // Clear previous results
    splQueryPre.textContent = '';
    llmQueryPre.textContent = '';
    systemPromptTextarea.value = '';
    totalAttemptsPre.textContent = '';
    errorMessageDiv.classList.add('hidden');
    errorText.textContent = '';
    resultDiv.style.display = 'none';
    if (vizRecommendation) {
        vizRecommendation.style.display = 'none';
    }

    try {
        const response = await fetch('/generate_query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query, llm_provider: llmProvider }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Format and display the SPL query
        const formattedQuery = formatSPLQuery(data.spl_query);
        splQueryPre.textContent = formattedQuery;

        // Display the LLM user query if available
        if (data.llm_query) {
            llmQueryPre.textContent = data.llm_query;
        }

        // Display the system prompt if available
        if (data.system_prompt) {
            systemPromptTextarea.value = data.system_prompt;
        }

        // Display total attempts if available
        if (data.total_attempts !== undefined) {
            totalAttemptsPre.textContent = data.total_attempts;
        }

        // Update visualization
        if (data.viz_type) {
            updateVisualization(data.viz_type);
        }

        resultDiv.style.display = 'block';
    } catch (error) {
        console.error('Error:', error);
        splQueryPre.textContent = '';
        llmQueryPre.textContent = '';
        systemPromptTextarea.value = '';
        totalAttemptsPre.textContent = '';
        
        // Display error message
        errorMessageDiv.classList.remove('hidden');
        errorText.textContent = error.message;
        
        resultDiv.style.display = 'block';
        if (vizRecommendation) {
            vizRecommendation.style.display = 'none';
        }
    }
}

// Function to format SPL query
function formatSPLQuery(query) {
    // Split the query by pipe character
    const parts = query.split(' | ');
    
    // Trim each part and join with newline and pipe
    return parts.map(part => part.trim()).join(' |\n    ');
}

// Function to handle visualization display
function updateVisualization(vizType) {
    const vizIcon = document.getElementById('viz-icon');
    const vizText = document.getElementById('viz-text');
    const vizRecommendation = document.getElementById('viz-recommendation');

    const vizConfig = {
        'table': { icon: '📋', text: 'Table View', color: '#3B82F6' },
        'chart': { icon: '📈', text: 'Chart Visualization', color: '#10B981' },
        '3D': { icon: '🌐', text: '3D Scatter Plot', color: '#8B5CF6' },
        'map': { icon: '🗺️', text: 'Geographic Map', color: '#EF4444' }
    };

    const { icon, text, color } = vizConfig[vizType] || { icon: '❓', text: 'Custom Visualization', color: '#6B7280' };
    
    if (vizIcon && vizText && vizRecommendation) {
        vizIcon.textContent = icon;
        vizText.textContent = text;
        vizRecommendation.style.borderLeft = `4px solid ${color}`;
        vizRecommendation.style.display = 'flex';
        vizRecommendation.classList.add('viz-update');
        setTimeout(() => {
            vizRecommendation.classList.remove('viz-update');
        }, 500);
    }
}

// Attach event listener to the generate button
document.addEventListener('DOMContentLoaded', function() {
    const generateBtn = document.getElementById('generate-btn');
    const queryInput = document.getElementById('query');

    if (generateBtn) {
        generateBtn.addEventListener('click', generateQuery);
    }

    if (queryInput) {
        // Allow generation on Enter key press in the query input
        queryInput.addEventListener('keypress', function(event) {
            if (event.key === 'Enter' && event.ctrlKey) {
                event.preventDefault();
                generateQuery();
            }
        });

        // Add tooltip to explain Ctrl+Enter shortcut
        queryInput.title = "Press Ctrl+Enter to generate query";
    }
});