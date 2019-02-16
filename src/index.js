import React from 'react';
import ReactDOM from 'react-dom';
import VideoStream from './components/video-stream';

class App extends React.Component {
  render() {
    return(
      <div>
        <VideoStream/>
      </div>
    );
  }
}

const rootElement = document.getElementById('root')
ReactDOM.render(<App />, rootElement)
