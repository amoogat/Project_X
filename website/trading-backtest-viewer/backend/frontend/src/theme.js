import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#29b6f6', // A light blue shade
    },
    secondary: {
      main: '#ffee58', // A yellow shade for highlighting important buttons or information
    },
    background: {
      default: '#102027', // A deep blue-grey
      paper: '#37474f',
    },
    text: {
      primary: '#eceff1', // Off-white text for readability on dark backgrounds
      secondary: '#cfd8dc',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h5: {
      fontWeight: 600,
      fontSize: '1.4rem',
      color: '#ffffff',
    },
    button: {
      textTransform: 'none', // Buttons with normal case text
      fontWeight: 500,
    },
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        colorPrimary: {
          backgroundColor: '#263238', // Darker shade for the AppBar
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        containedPrimary: {
          color: 'white', // Ensuring text on primary buttons is always white
        },
      },
    },
  },
});

export default theme;
