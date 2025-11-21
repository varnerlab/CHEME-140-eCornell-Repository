


# --- PUBLIC METHODS BELOW HERE -------------------------------------------------------------------------------- #
"""
    build(base::String, model::MyWeatherGridPointEndpointModel) -> String

This function is used to build a URL string that can be used to make a HTTP GET call to the National Weather Service API.
It takes two arguments, a base URL string, and a model of type `MyWeatherGridPointEndpointModel`. 

### Arguments
- `base::String` - The base URL string.
- `model::MyWeatherGridPointEndpointModel` - The model that contains the latitude and longitude of the grid point.

### Returns
- `String` - The complete URL string that can be used to make a HTTP GET call to the National Weather Service API.
"""
function build(base::String, model::MyWeatherGridPointEndpointModel)::String
    
    # TODO: implement this function, and remove the throw statement
    # throw(ArgumentError("build(base::String, model::MyWeatherGridPointEndpointModel) not implemented yet!"));

    # build the URL string -
    url_string = "$(base)/points/$(model.latitude),$(model.longitude)";

    # return the URL string -
    return url_string;
end

function build(base::String, model::MyBiggModelsEndpointModel; apiversion::String = "v2")::String
    
    # TODO: implement this function, and remove the throw statement
    # throw(ArgumentError("build(base::String, model::MyWeatherGridPointEndpointModel) not implemented yet!"));

    # build the URL string -
    url_string = "$(base)/api/$(apiversion)/models";

    # return the URL string -
    return url_string;
end

function build(base::String, model::MyBiggModelsDownloadModelEndpointModel; apiversion::String = "v2")::String

    # get data -
    bigg_id = model.bigg_id;

    # build the URL string -
    url_string = "$(base)/api/$(apiversion)/models/$(bigg_id)/download";

    # return the URL string -
    return url_string;
end
# --- PUBLIC METHODS ABOVE HERE -------------------------------------------------------------------------------- #