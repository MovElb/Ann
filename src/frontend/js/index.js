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
