export default function RootLayout({ children }) {
    return (
      <html lang="en">
        <head>
          <meta charSet="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>Conversational Healthcare Chatbot</title>
          {/* Add any other global meta tags, scripts, or stylesheets here */}
        </head>
        <body>
          {children}
        </body>
      </html>
    );
  }
  