import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    background: {
      default: '#303030',  // Darker background for a modern look
      paper: '#424242',    // Mid grey for components
    },
    primary: {
      main: '#90caf9',     // Light blue for primary actions
    },
    secondary: {
      main: '#c4afj1',     // Soft pink for secondary actions
    },
    text: {
      primary: '#ffffff',  // White text for high contrast
      secondary: '#eeeeee' // Light grey for less important text
    }
  },
  typography: {
    fontFamily: 'Roboto, sans-serif',
    h5: {
      color: '#90caf9',   // Use primary color for headers
      fontWeight: 'bold'
    },
    body2: {
      color: '#eeeeee',   // Light grey for secondary text
    }
  }
});

export default theme;
