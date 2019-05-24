// disable context field if checkbox not checked
// set required attribute as true if checkbox is checked
var chk = document.getElementById("custom_context_id");
var context = document.getElementById("context_id");
chk.addEventListener("click", enable);
function enable() {
    context.disabled = !this.checked;
    context.required = true;
}

// replace validation bubble with red warning
function replaceValidationUI( form ) {
    form.addEventListener( "invalid", function( event ) {
        event.preventDefault();
    }, true );
    form.addEventListener( "submit", function( event ) {
        if ( !this.checkValidity() ) {
            event.preventDefault();
        }
    });
    var submitButton = form.querySelector( "button:not([type=button]), input[type=submit]" );
    submitButton.addEventListener( "click", function( event ) {
        var invalidFields = form.querySelectorAll( ":invalid" ),
            errorMessages = form.querySelectorAll( ".error-message" ),
            parent;
        for ( var i = 0; i < errorMessages.length; i++ ) {
            errorMessages[ i ].parentNode.removeChild( errorMessages[ i ] );
        }
        for ( var i = 0; i < invalidFields.length; i++ ) {
            parent = invalidFields[ i ].parentNode;
            parent.insertAdjacentHTML( "beforeend", "<div class='error-message'>" +
                invalidFields[ i ].validationMessage +
                "</div>" );
        }
        if ( invalidFields.length > 0 ) {
            invalidFields[ 0 ].focus();
        }
    });
}
var forms = document.querySelectorAll( "form" );
for ( var i = 0; i < forms.length; i++ ) {
    replaceValidationUI( forms[ i ] );
}

// send json request
const isValidElement = element => {
  return element.name && element.value && !element.disabled;
};
const isValidValue = element => {
  return (!['checkbox', 'radio'].includes(element.type) || element.checked);
};
const formToJSON = elements => [].reduce.call(elements, (data, element) => {
  if (isValidElement(element) && isValidValue(element)) {
      data[element.name] = element.value;
  }
  return data;
}, {});

const handleFormSubmit = event => {
  event.preventDefault();
  const data = formToJSON(form.elements);

  // Demo only: print the form data onscreen as a formatted JSON object.
  const dataContainer = document.getElementsByClassName('results__display')[0];
  dataContainer.textContent = JSON.stringify(data, null, "  ");

  // var xhr = new XMLHttpRequest()
  // if(xhr)
  //   {
  //     xhr.open('POST', https://seann.ru/api/search, true);
  //     // xhr.setRequestHeader('X-PINGOTHER', 'pingpong');
  //     xhr.setRequestHeader('Content-Type', 'application/json');
  //     // xhr.onreadystatechange = handler;
  //     xhr.send(JSON.stringify(data));
  //   }
  fetch('https://seann.ru/api/search', {
    method: 'post',
    mode: 'cors',
    withCredentials: false,
    headers: {
      "Content-type": "application/json"
    },
    body: JSON.stringify(data)
  })
  .then(function (json) {
    console.log('Request succeeded with JSON response', json);
    // const dataContainer = document.getElementsByClassName('results__display')[0];
    dataContainer.textContent = JSON.stringify(json, null, "  ");
  })
  .catch(function (error) {
    console.log('Request failed', error);
  });
};

const form = document.getElementById( "custom_request_id" );
form.addEventListener('submit', handleFormSubmit);
// dlers.
//   xhr.onload = function() {
//     var text = xhr.responseText;
//     var title = getTitle(text);
//     alert('Response from CORS request to ' + url + ': ' + title);
//   };
//
//   xhr.onerror = function() {
//     alert('Woops, there was an error making the request.');
//   };
//
//   xhr.send(data);
// }
