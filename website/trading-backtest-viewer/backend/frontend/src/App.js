import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import UploadForm from './components/UploadForm';
import { AppBar, Toolbar, Typography, Button, Container } from '@mui/material';
import theme from './theme'; // Import the theme
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline, GlobalStyles } from '@mui/material';
function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline /> {/* This resets padding and margins and applies background.default to the body */}
        <GlobalStyles
          styles={{
            body: { backgroundColor: theme.palette.background.default },
            '#root': { height: '100vh', backgroundColor: theme.palette.background.default }
          }}
        />
        <Router>
        <div>
            <AppBar position="static">
            <Toolbar>
                <Typography variant="h6" style={{ flexGrow: 1 }}>
                Trading Backtest Viewer
                </Typography>
                <Button color="inherit" component={Link} to="/upload">Upload</Button>
                <Button color="inherit" component={Link} to="/results">Results</Button>
            </Toolbar>
            </AppBar>
            <Container style={{ backgroundColor: '#1a42c', color: 'white' }}>
            <Routes>
                <Route path="/upload" element={<UploadForm />} />
                <Route path="/results" element={<div>Results Page</div>} /> {/* Placeholder for Results page */}
                <Route path="/" element={<UploadForm />} />
            </Routes>
            </Container>
        </div>
        </Router>
    </ThemeProvider>
  );
}

export default App;
