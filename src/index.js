import React from 'react';
import ReactDOM from 'react-dom';
import VideoStream from './components/video-stream';
import {Helmet} from 'react-helmet';

class App extends React.Component {
  render() {
    return(
      <div className="body">
        <Helmet>
            <style>{'body { background-color: #01E174; }'}</style>
          </Helmet>
        <VideoStream/>
      </div>
    );
  }
}

const rootElement = document.getElementById('root')
ReactDOM.render(<App />, rootElement)
