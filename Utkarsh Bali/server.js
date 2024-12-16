const express = require('express');
const OpenAI = require('openai');
const pdfParse = require('pdf-parse');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Set OpenAI API key
const OPENAI_API_KEY = '';
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// Global context and rules
let context = [];
const rules = `
You are an iris dataset assistant at Purdue University.
Be cordial at all times.
Answer any questions that customers have regarding the iris dataset.
When the user asks you to calculate or find something, just give the final value. Be clear and concise.
Based on the iris_bot_training_pdf.pdf and irisDataRaw that you have access to, calculate and answer any questions that the customer asks.
Use this function to calculate mean: function calculateMean(feature) {
const values = irisData.map(row => row[feature]).filter(val => !isNaN(val));
const sum = values.reduce((acc, val) => acc + val, 0);
return sum / values.length;
}

You can calculate SHAP (SHapley Additive exPlanations) values for the Iris dataset.
To request SHAP values, use the format: "Can you calculate SHAP values for SepalLengthCm=X, SepalWidthCm=Y, PetalLengthCm=Z, PetalWidthCm=W?"
Replace X, Y, Z, and W with the actual numeric values for the features.
If a user asks for a mean calculation of any feature, inform them that you'll use a custom function to calculate it accurately, and then use the iris dataset in iris_bot_training_pdf.pdf to calculate and give user the calculation and clearly state the final value.
`;

// Iris dataset
const irisDataRaw = `Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
1,5.1,3.5,1.4,0.2,Iris-setosa
2,4.9,3.0,1.4,0.2,Iris-setosa
`;

// Convert raw data to structured format
const irisData = irisDataRaw.split('\n').slice(1).map((row, index) => {
    const [Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species] = row.split(',');
    if (!Id || !SepalLengthCm || !SepalWidthCm || !PetalLengthCm || !PetalWidthCm) {
      console.error(`Malformed data at row ${index + 2}: ${row}`);
    }
    return {
      Id: parseInt(Id) || 0,
      SepalLengthCm: parseFloat(SepalLengthCm) || 0,
      SepalWidthCm: parseFloat(SepalWidthCm) || 0,
      PetalLengthCm: parseFloat(PetalLengthCm) || 0,
      PetalWidthCm: parseFloat(PetalWidthCm) || 0,
      Species: Species ? Species.trim() : 'Unknown'
    };
  }).filter(item => item.Id !== 0);

// Custom function to calculate mean
function calculateMean(feature) {
  const values = irisData.map(row => row[feature]).filter(val => !isNaN(val));
  const sum = values.reduce((acc, val) => acc + val, 0);
  return sum / values.length;
}

// Function to extract feature name from user input
function extractFeatureName(input) {
  const features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'];
  for (let feature of features) {
    if (input.toLowerCase().includes(feature.toLowerCase())) {
      return feature;
    }
  }
  return null;
}

// Read PDF and extract its text
async function readPdf(filePath) {
  try {
    const dataBuffer = fs.readFileSync(filePath);
    const pdfData = await pdfParse(dataBuffer);
    return pdfData.text;
  } catch (error) {
    console.error('Error reading PDF:', error);
    throw new Error('Failed to read PDF.');
  }
}

// Append initial system context with PDF data and rules
async function initContext() {
  try {
    const pdfText = await readPdf('iris_bot_training_pdf.pdf');
    context.push({
      role: 'system',
      content: `${rules} ${pdfText}`
    });
    console.log('Context initialized with PDF data and rules.');
  } catch (error) {
    console.error('Error initializing context:', error);
    throw error;
  }
}

// Function to process user input and get GPT response
async function processUserInput(userInput) {
  try {
    console.log(`User input: ${userInput}`);
    
    // Add user input to context
    context.push({ role: 'user', content: userInput });

    // Get GPT response
    const completion = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: context,
    });

    const response = completion.choices[0].message.content;
    console.log('GPT response:', response);

    // Add GPT response to context
    context.push({ role: 'assistant', content: response });

    return response;
  } catch (error) {
    console.error('Error processing user input:', error);
    throw new Error('Failed to process input.');
  }
}

// Middleware to parse JSON requests
app.use(express.json());
app.use(express.static('public'));

// Serve index.html for the root URL
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// POST route to handle user questions
app.post('/chat', async (req, res) => {
  const { userInput } = req.body;
  if (!userInput) {
    console.error('No user input provided.');
    return res.status(400).json({ error: 'No question provided.' });
  }

  try {
    const response = await processUserInput(userInput);
    res.json({ response });
  } catch (error) {
    console.error('Error processing user input:', error);
    res.status(500).json({ error: 'Failed to process the request.' });
  }
});

// Initialize context and start server
initContext().then(() => {
  app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
  });
}).catch(error => {
  console.error('Error initializing context:', error);
});
