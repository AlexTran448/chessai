import React from 'react';
import { Button } from "../../../../../OneDrive/Documents/GitHub/thirdyearchess/frontend/registry/default/ui/button"

class UpdateHomographyButton extends React.Component {
    constructor(props: {} | Readonly<{}>) {
        super(props);
        this.handleClick = this.handleClick.bind(this);
    }

    handleClick() {
        // Call the update_homography function from your backend here
        fetch("/api/boardaligner/update_homography", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
        }).then((res) => res.json())
    }

    render() {
        return (
            <Button onClick={this.handleClick}>
                Update Homography
            </Button>
        );
    }
}

export default UpdateHomographyButton;
